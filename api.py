from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import tempfile
from typing import List
from extractor import SmartExtractor

app = FastAPI()

# Permite CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = SmartExtractor(use_session_cache=True)

@app.post("/api/extract")
async def extract_document(
    file: UploadFile = File(...),
    label: str = Form(...),
    extraction_schema: str = Form(...)
):
    """Extrai dados de um documento"""
    
    # Parse schema JSON
    schema = json.loads(extraction_schema)
    
    # Salva arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Processa documento
        result = extractor.extract(label, schema, tmp_path)
        
        return {
            "success": True,
            "data": {
                "label": label,
                "fileName": file.filename,
                "extractionTime": result.extraction_time,
                "cost": result.total_cost,
                "llmCalls": result.llm_calls,
                "fromCache": result.extraction_time == 0.0,
                "fields": {
                    name: {
                        "value": field.value,
                        "confidence": field.confidence,
                        "method": field.method
                    }
                    for name, field in result.fields.items()
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        Path(tmp_path).unlink()

@app.post("/api/extract-batch")
async def extract_batch(
    files: List[UploadFile] = File(...),
    labels: str = Form(...),
    schemas: str = Form(...)
):
    """Extrai dados de múltiplos documentos"""
    
    labels_list = json.loads(labels)
    schemas_list = json.loads(schemas)
    
    results = []
    
    for file, label, schema in zip(files, labels_list, schemas_list):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = extractor.extract(label, schema, tmp_path)
            
            results.append({
                "success": True,
                "fileName": file.filename,
                "label": label,
                "extractionTime": result.extraction_time,
                "cost": result.total_cost,
                "llmCalls": result.llm_calls,
                "fromCache": result.extraction_time == 0.0,
                "fields": {
                    name: {
                        "value": field.value,
                        "confidence": field.confidence,
                        "method": field.method
                    }
                    for name, field in result.fields.items()
                }
            })
        except Exception as e:
            results.append({
                "success": False,
                "fileName": file.filename,
                "error": str(e)
            })
        finally:
            Path(tmp_path).unlink()
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)