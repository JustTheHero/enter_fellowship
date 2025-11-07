import os
import re
import json
import fitz  # PyMuPDF
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from openai import OpenAI

# Configura o cliente da OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Mantém logs de erros
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Define a estrutura de diretórios
BASE_DIR = Path("./document_learning")
SCHEMAS_DIR = BASE_DIR / "schemas"
TEMPLATES_DIR = BASE_DIR / "templates"
RAG_DIR = BASE_DIR / "rag"

# Garante que os diretórios existam
for directory in [SCHEMAS_DIR, TEMPLATES_DIR, RAG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class FieldExtraction:
    # Estrutura para um único campo extraído
    value: Optional[str]
    confidence: float
    method: str  # Ex: "pattern", "rag", "llm"
    context: str
    position: Optional[Tuple[int, int]] = None
    cost: float = 0.0

@dataclass
class ExtractionResult:
    # Estrutura para o resultado completo da extração
    label: str
    fields: Dict[str, FieldExtraction]
    extraction_time: float
    schema_version: int
    total_cost: float
    llm_calls: int

class LocalRAG:
    # Aprende e reutiliza extrações de documentos similares
    
    def __init__(self, rag_dir: Path = RAG_DIR):
        self.rag_dir = rag_dir
        self.examples: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.max_examples_per_field = 15
        self._load_examples()
    
    def _load_examples(self):
        # Carrega os arquivos .pkl do RAG para a memória
        for rag_file in self.rag_dir.glob("*.pkl"):
            try:
                with open(rag_file, 'rb') as f:
                    label = rag_file.stem
                    self.examples[label] = pickle.load(f)
            except Exception as e:
                logging.error(f"Erro ao carregar RAG {rag_file}: {e}")
    
    def add_example(self, label: str, field_name: str, value: str, 
                    context: str, full_text: str, confidence: float = 1.0):
        # Salva uma nova extração para reutilizar depois
        example = {
            "value": value,
            "context": context[:300],
            "text_sample": full_text[:500],
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "text_hash": hashlib.md5(full_text.encode()).hexdigest()[:8]
        }
        
        # Evita salvar duplicatas
        existing_hashes = {e['text_hash'] for e in self.examples[label][field_name]}
        if example['text_hash'] in existing_hashes:
            return
        
        self.examples[label][field_name].append(example)
        
        # Mantém apenas os N melhores exemplos
        self.examples[label][field_name] = sorted(
            self.examples[label][field_name],
            key=lambda x: (x['confidence'], x['timestamp']),
            reverse=True
        )[:self.max_examples_per_field]
        
        self._save_examples(label)
    
    def find_similar_examples(self, label: str, field_name: str, 
                              current_text: str, k: int = 5) -> List[Dict]:
        # Encontra os 'k' exemplos mais similares ao texto atual
        if label not in self.examples or field_name not in self.examples[label]:
            return []
        
        examples = self.examples[label][field_name]
        text_sample = current_text[:500]
        
        scored_examples = []
        for example in examples:
            # Calcula a similaridade
            similarity = SequenceMatcher(
                None, 
                text_sample.lower(), 
                example['text_sample'].lower()
            ).ratio()
            scored_examples.append({**example, 'similarity': similarity})
        
        scored_examples.sort(key=lambda x: (x['similarity'], x['confidence']), reverse=True)
        return scored_examples[:k]
    
    def extract_from_rag(self, label: str, field_name: str, 
                         current_text: str) -> Tuple[Optional[str], float, str]:
        # Tenta extrair um valor baseado em exemplos similares
        similar_examples = self.find_similar_examples(label, field_name, current_text, k=3)
        
        if not similar_examples:
            return None, 0.0, ""
        
        best_example = similar_examples[0]
        
        # Só usa o RAG se a similaridade for alta
        if best_example['similarity'] > 0.80:
            example_context = best_example['context']
            example_value = best_example['value']
            keywords = self._extract_keywords_from_context(example_context, example_value)
            
            for keyword in keywords:
                # Tenta achar a mesma palavra-chave no *novo* documento
                pattern = rf"{re.escape(keyword)}\s*:?\s*([^\n]{{0,150}}?)(?:\n|$)"
                match = re.search(pattern, current_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    value = re.sub(r'[_\-\.]{3,}', '', value).strip()
                    if len(value) > 2:
                        confidence = best_example['similarity'] * 0.9
                        return value, confidence, match.group(0)
        
        return None, 0.0, ""
    
    def _extract_keywords_from_context(self, context: str, value: str) -> List[str]:
        # Pega as últimas palavras significativas antes do valor
        value_idx = context.find(value)
        if value_idx == -1: return []
        
        before_text = context[:value_idx]
        words = re.findall(r'[A-Za-zÀ-ÿ]{4,}', before_text)
        return words[-3:] if len(words) >= 3 else words
    
    def _save_examples(self, label: str):
        # Serializa os exemplos em um arquivo pickle
        rag_path = self.rag_dir / f"{label}.pkl"
        with open(rag_path, 'wb') as f:
            pickle.dump(dict(self.examples[label]), f)

class MinimalLLMExtractor:
    # Classe que usa um LLM (ex: GPT) como fallback
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.cost_per_1k_input = 0.00015
        self.cost_per_1k_output = 0.0006
    
    def batch_extract_missing_fields(self, missing_fields: Dict[str, str], 
                                     text: str, 
                                     rag_examples: Dict[str, List[Dict]] = None) -> Dict[str, Tuple[str, float, float]]:
        # Extrai múltiplos campos de uma só vez usando o LLM
        if not missing_fields:
            return {}
        
        rag_examples = rag_examples or {}
        examples_text = self._format_rag_examples(rag_examples)
        fields_list = "\n".join([f"- {name}: {desc}" for name, desc in missing_fields.items()])
        
        # Monta o prompt para a extração em lote
        prompt = f"""Extract the following fields from the document. Return ONLY valid JSON.
FIELDS TO EXTRACT:
{fields_list}
{examples_text}
DOCUMENT TEXT:
{text[:4000]}
CRITICAL INSTRUCTIONS:
- Return JSON format: {{"field": "exact_value"}}
- Use null ONLY if field does NOT exist in document
- Extract EXACTLY as shown in document
JSON:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            results = json.loads(json_match.group(0)) if json_match else {}
            
            tokens_input = response.usage.prompt_tokens
            tokens_output = response.usage.completion_tokens
            cost = (tokens_input / 1000 * self.cost_per_1k_input + 
                    tokens_output / 1000 * self.cost_per_1k_output)
            
            processed_results = {}
            for field_name in missing_fields.keys():
                value = results.get(field_name)
                if value and str(value).lower() != "null":
                    processed_results[field_name] = (str(value), 0.85, cost / max(len(missing_fields), 1))
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Erro LLM: {e}")
            return {}
    
    def _format_rag_examples(self, rag_examples: Dict[str, List[Dict]]) -> str:
        # Formata os exemplos do RAG para incluir no prompt
        if not rag_examples or not any(rag_examples.values()):
            return ""
        
        examples_text = "PREVIOUS EXTRACTION EXAMPLES (from similar documents):\n"
        for field_name, examples in rag_examples.items():
            if examples:
                best_example = examples[0]
                examples_text += f"\nField '{field_name}': {best_example['value']}\n"
        
        return examples_text

class DynamicSchemaManager:
    # Gerencia os "schemas" (tipos de documentos e seus campos)
    
    def __init__(self, schemas_dir: Path = SCHEMAS_DIR):
        self.schemas_dir = schemas_dir
        self.schemas: Dict[str, Dict] = {}
        self._load_all_schemas()
    
    def _load_all_schemas(self):
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                    self.schemas[schema_data['label']] = schema_data
            except Exception as e:
                logging.error(f"Erro ao carregar schema {schema_file}: {e}")
    
    def get_or_create_schema(self, label: str) -> Dict:
        # Se o schema não existir, cria um novo
        if label not in self.schemas:
            self.schemas[label] = {
                "label": label, "version": 1,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "fields": {}, "total_documents_seen": 0
            }
        return self.schemas[label]
    
    def update_schema(self, label: str, new_fields: Dict[str, str]):
        # Atualiza um schema com novos campos
        schema = self.get_or_create_schema(label)
        
        for field_name, field_description in new_fields.items():
            if field_name not in schema['fields']:
                schema['fields'][field_name] = {
                    "description": field_description,
                    "first_seen": datetime.now().isoformat(),
                    "occurrences": 0
                }
            schema['fields'][field_name]['occurrences'] += 1
        
        schema['version'] += 1
        schema['updated_at'] = datetime.now().isoformat()
        schema['total_documents_seen'] += 1
        self._save_schema(label)
        return schema
    
    def _save_schema(self, label: str):
        # Salva o schema atualizado no arquivo JSON
        schema_path = self.schemas_dir / f"{label}.json"
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(self.schemas[label], f, indent=2, ensure_ascii=False)

class DynamicTemplateAnalyzer:
    # Aprende onde (keywords, regex) os campos aparecem
    
    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.templates: Dict[str, Dict] = {}
        self._load_templates()
    
    def _load_templates(self):
        for template_file in self.templates_dir.glob("*.pkl"):
            try:
                with open(template_file, 'rb') as f:
                    self.templates[template_file.stem] = pickle.load(f)
            except Exception as e:
                logging.error(f"Erro ao carregar template: {e}")
    
    def learn_field_position(self, label: str, field_name: str, 
                             text: str, value: str, context: str):
        # Aprende a posição de um campo com base em uma extração
        if label not in self.templates:
            self.templates[label] = {"field_positions": {}, "examples_seen": 0}
        
        template = self.templates[label]
        
        if field_name not in template['field_positions']:
            template['field_positions'][field_name] = {"keywords": set(), "regex_patterns": set()}
        
        field_data = template['field_positions'][field_name]
        
        # Aprende palavras-chave que vêm *antes* do valor
        keywords = self._extract_keywords(text, value)
        field_data['keywords'].update(keywords)
        
        if self._has_pattern(value):
            pattern = self._generate_pattern(value)
            if pattern:
                field_data['regex_patterns'].add(pattern)
        
        template['examples_seen'] += 1
        self._save_template(label)
    
    def _extract_keywords(self, text: str, value: str) -> List[str]:
        idx = text.find(value)
        if idx == -1: return []
        
        before = text[max(0, idx-50):idx]
        words = re.findall(r'[A-Za-zÀ-ÿ]{3,}', before)
        return words[-3:] if words else []
    
    def _has_pattern(self, value: str) -> bool:
        # Regex para padrões comuns (CPF, data, CNPJ, etc.)
        patterns = [
            r'\d{3}\.\d{3}\.\d{3}-\d{2}', r'\d{2}/\d{2}/\d{4}',
            r'\d{5}-\d{3}', r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}',
        ]
        return any(re.match(p, value) for p in patterns)
    
    def _generate_pattern(self, value: str) -> Optional[str]:
        # Generaliza o padrão (ex: '123' -> '\d\d\d')
        pattern = re.sub(r'\d', r'\\d', value)
        pattern = re.sub(r'[A-Z]', r'[A-Z]', pattern)
        return pattern
    
    def get_hints(self, label: str, field_name: str) -> Dict:
        if label not in self.templates or field_name not in self.templates[label]['field_positions']:
            return {}
        
        # Retorna as "dicas" (keywords e regex) aprendidas
        field_data = self.templates[label]['field_positions'][field_name]
        return {
            "keywords": list(field_data['keywords']),
            "regex_patterns": list(field_data['regex_patterns'])
        }
    
    def _save_template(self, label: str):
        template_path = self.templates_dir / f"{label}.pkl"
        template = self.templates[label].copy()
        
        # Converte 'set' para 'list' para salvar em pickle
        for field_data in template.get('field_positions', {}).values():
            field_data['keywords'] = list(field_data['keywords'])
            field_data['regex_patterns'] = list(field_data['regex_patterns'])
        
        with open(template_path, 'wb') as f:
            pickle.dump(template, f)

class SmartExtractor:
    # Classe principal que orquestra a extração e o aprendizado
    
    def __init__(self):
        self.schema_manager = DynamicSchemaManager()
        self.template_analyzer = DynamicTemplateAnalyzer()
        self.rag = LocalRAG()
        self.llm = MinimalLLMExtractor()
        
        # Regex genéricas para extração de alta confiança
        self.common_patterns = {
            "cpf": r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b',
            "cnpj": r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b',
            "rg": r'\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b',
            "cep": r'\b\d{5}-?\d{3}\b',
            "data": r'\b\d{2}[/\-\.]\d{2}[/\-\.]\d{4}\b',
            "telefone": r'\(?\d{2}\)?\s*\d{4,5}-?\d{4}',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "valor": r'R\$\s*[\d.,]+'
        }
    
    def extract(self, label: str, extraction_schema: Dict[str, str], 
                pdf_path: str) -> ExtractionResult:
        # Estratégia em cascata: Pattern -> Template -> RAG -> Keyword -> LLM
        
        start_time = datetime.now()
        text = self._extract_text_from_pdf(pdf_path)
        self.schema_manager.update_schema(label, extraction_schema)
        
        field_results = {}
        missing_fields = {}
        total_cost = 0.0
        
        for field_name, field_description in extraction_schema.items():
            # Tenta extrair por Regex de padrão comum (ex: CPF)
            extraction = self._extract_by_pattern(field_name, text)
            
            # Tenta extrair por Template aprendido
            if not extraction.value:
                extraction = self._extract_by_template(label, field_name, text)
            
            # Tenta extrair por RAG
            if not extraction.value:
                extraction = self._extract_by_rag(label, field_name, text)
            
            # Tenta extrair por Keyword (nome do campo)
            if not extraction.value:
                extraction = self._extract_by_keyword(field_name, field_description, text)
            
            if extraction.value:
                field_results[field_name] = extraction
                
                # Se a confiança for boa, aprende com essa extração
                if extraction.confidence > 0.6:
                    self.template_analyzer.learn_field_position(
                        label, field_name, text, extraction.value, extraction.context
                    )
                    self.rag.add_example(
                        label, field_name, extraction.value, 
                        extraction.context, text, extraction.confidence
                    )
            else:
                missing_fields[field_name] = field_description
        
        llm_calls = 0
        # Fallback: Agrupa campos faltantes e usa LLM
        if missing_fields:
            rag_examples = {
                field: self.rag.find_similar_examples(label, field, text, k=2)
                for field in missing_fields.keys()
            }
            
            llm_results = self.llm.batch_extract_missing_fields(
                missing_fields, text, rag_examples
            )
            llm_calls = 1 if llm_results else 0
            
            for field_name, (value, confidence, cost) in llm_results.items():
                field_results[field_name] = FieldExtraction(
                    value=value, confidence=confidence, method="llm",
                    context=text[:200], cost=cost
                )
                total_cost += cost
                
                # Aprende também com o resultado do LLM
                self.rag.add_example(
                    label, field_name, value, text[:200], text, confidence
                )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResult(
            label=label,
            fields=field_results,
            extraction_time=duration,
            schema_version=self.schema_manager.schemas[label]['version'],
            total_cost=total_cost,
            llm_calls=llm_calls
        )
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        # Usa PyMuPDF (fitz) para extrair texto da primeira página
        doc = fitz.open(pdf_path)
        text = doc[0].get_text("text")
        doc.close()
        return text
    
    def _extract_by_pattern(self, field_name: str, text: str) -> FieldExtraction:
        # Extração por regex de padrões comuns (alta confiança)
        field_lower = field_name.lower()
        
        for pattern_name, pattern in self.common_patterns.items():
            if pattern_name in field_lower:
                match = re.search(pattern, text)
                if match:
                    value = match.group(0)
                    context = text[max(0, match.start()-50):match.end()+50]
                    return FieldExtraction(
                        value=value, confidence=0.95, method="pattern",
                        context=context, position=(match.start(), match.end())
                    )
        
        return FieldExtraction(None, 0.0, "not_found", "")
    
    def _extract_by_template(self, label: str, field_name: str, text: str) -> FieldExtraction:
        # Extração baseada no 'DynamicTemplateAnalyzer'
        hints = self.template_analyzer.get_hints(label, field_name)
        if not hints:
            return FieldExtraction(None, 0.0, "not_found", "")
        
        for pattern in hints.get('regex_patterns', []):
            try:
                match = re.search(pattern, text)
                if match:
                    value = match.group(0)
                    context = text[max(0, match.start()-50):match.end()+50]
                    return FieldExtraction(value, 0.90, "template", context)
            except:
                continue
        
        for keyword in hints.get('keywords', []):
            pattern = rf"{re.escape(keyword)}\s*:?\s*([^\n]{{0,100}}?)(?:\n|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = re.sub(r'[_\-\.]{3,}', '', value).strip()
                if len(value) > 2:
                    return FieldExtraction(value, 0.85, "template", match.group(0))
        
        return FieldExtraction(None, 0.0, "not_found", "")
    
    def _extract_by_rag(self, label: str, field_name: str, text: str) -> FieldExtraction:
        # Deixa a lógica para a classe RAG
        value, confidence, context = self.rag.extract_from_rag(label, field_name, text)
        if value:
            return FieldExtraction(value, confidence, "rag", context)
        return FieldExtraction(None, 0.0, "not_found", "")
    
    def _extract_by_keyword(self, field_name: str, field_description: str, text: str) -> FieldExtraction:
        # Heurística simples: busca o nome do campo no texto
        keywords = [field_name.lower(), field_name.replace('_', ' ').lower()]
        
        for keyword in keywords:
            pattern = rf"{re.escape(keyword)}\s*:?\s*([^\n]{{5,100}}?)(?:\n|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = re.sub(r'[_\-\.]{3,}', '', value).strip()
                if len(value) > 2:
                    return FieldExtraction(value, 0.70, "keyword", match.group(0))
        
        return FieldExtraction(None, 0.0, "not_found", "")