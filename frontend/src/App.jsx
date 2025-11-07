import React, { useState } from 'react';
import { Upload, FileText, Plus, Trash2, Play, Download, AlertCircle, Clock, DollarSign } from 'lucide-react';

const DocumentExtractorUI = () => {
  const [documents, setDocuments] = useState([]);
  const [currentLabel, setCurrentLabel] = useState('');
  const [schemaFields, setSchemaFields] = useState([{ name: '', description: '' }]);
  const [results, setResults] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    const newDocs = files.map(file => ({
      id: Date.now() + Math.random(),
      file: file,
      label: currentLabel || 'documento',
      schema: schemaFields.filter(f => f.name && f.description),
      status: 'pending'
    }));
    setDocuments([...documents, ...newDocs]);
  };

  const addSchemaField = () => setSchemaFields([...schemaFields, { name: '', description: '' }]);
  const removeSchemaField = (i) => setSchemaFields(schemaFields.filter((_, idx) => idx !== i));
  const updateSchemaField = (i, field, value) => {
    const updated = [...schemaFields];
    updated[i][field] = value;
    setSchemaFields(updated);
  };
  const removeDocument = (id) => setDocuments(documents.filter(doc => doc.id !== id));
  const resetForm = () => {
    setCurrentLabel('');
    setSchemaFields([{ name: '', description: '' }]);
  };

  const processDocuments = async () => {
    if (!documents.length) {
      setError('Adicione pelo menos um documento');
      return;
    }

    setProcessing(true);
    setError('');
    const processedResults = [];

    for (let doc of documents) {
      setDocuments(prev => prev.map(d => d.id === doc.id ? { ...d, status: 'processing' } : d));

      const formData = new FormData();
      formData.append('file', doc.file);
      formData.append('label', doc.label);
      formData.append('extraction_schema', JSON.stringify(
        doc.schema.reduce((acc, f) => ({ ...acc, [f.name]: f.description }), {})
      ));

      try {
        const res = await fetch('http://localhost:8000/api/extract', { method: 'POST', body: formData });
        const result = await res.json();

        if (result.success) {
          processedResults.push({ id: doc.id, ...result.data });
          setDocuments(prev => prev.map(d => d.id === doc.id ? { ...d, status: 'completed' } : d));
        } else {
          throw new Error(result.error);
        }
      } catch (err) {
        setError(`Erro ao processar ${doc.file.name}: ${err.message}`);
        setDocuments(prev => prev.map(d => d.id === doc.id ? { ...d, status: 'error' } : d));
      }
    }

    setResults(processedResults);
    setProcessing(false);
  };

  const exportResults = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `extraction_results_${Date.now()}.json`;
    link.click();
  };

  const getStatusColor = (status) => ({
    pending: 'text-gray-500',
    processing: 'text-blue-600',
    completed: 'text-green-600',
    error: 'text-red-600'
  }[status] || 'text-gray-500');

  const getMethodColor = (method) => ({
    pattern: 'text-green-600',
    rag: 'text-blue-600',
    llm: 'text-purple-600',
    template: 'text-yellow-600',
    keyword: 'text-orange-600'
  }[method] || 'text-gray-600');

  return (
    <div className="min-h-screen bg-slate-50 p-8 text-slate-800">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8 border-b border-slate-200 pb-4">
          <h1 className="text-3xl font-bold text-slate-800">Extração de Documentos</h1>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
          {/* Configuração do Schema */}
          <section className="bg-white border border-slate-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Schema de Extração</h2>

            <label className="block text-sm font-medium mb-2">Label do Documento *</label>
            <input
              type="text"
              value={currentLabel}
              onChange={(e) => setCurrentLabel(e.target.value)}
              placeholder="Ex: nota_fiscal, contrato..."
              className="w-full border border-slate-300 rounded px-3 py-2 mb-4"
            />

            <h3 className="text-sm font-medium mb-2">Campos</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto mb-4">
              {schemaFields.map((f, i) => (
                <div key={i} className="p-3 border border-slate-200 rounded-lg">
                  <input
                    type="text"
                    value={f.name}
                    onChange={(e) => updateSchemaField(i, 'name', e.target.value)}
                    placeholder="nome_do_campo"
                    className="w-full border border-slate-300 rounded px-3 py-2 mb-2 text-sm"
                  />
                  <textarea
                    value={f.description}
                    onChange={(e) => updateSchemaField(i, 'description', e.target.value)}
                    placeholder="Descreva onde o campo aparece..."
                    className="w-full border border-slate-300 rounded px-3 py-2 text-sm h-20 resize-none"
                  />
                  <button onClick={() => removeSchemaField(i)} className="text-red-600 text-xs mt-2">
                    Remover campo
                  </button>
                </div>
              ))}
            </div>

            <button
              onClick={addSchemaField}
              className="w-full border border-slate-300 rounded py-2 text-sm hover:bg-slate-100 mb-3"
            >
              <Plus size={16} className="inline mr-1" /> Adicionar Campo
            </button>

            <button onClick={resetForm} className="w-full text-sm text-slate-600 hover:text-slate-800">
              Limpar formulário
            </button>
          </section>

          {/* Upload de Documentos */}
          <section className="bg-white border border-slate-200 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Upload de Documentos</h2>

            <label className="border border-dashed border-slate-300 rounded-lg p-8 block cursor-pointer hover:border-blue-400">
              <input type="file" multiple accept=".pdf" onChange={handleFileUpload} className="hidden" />
              <div className="text-center text-slate-600">
                <Upload size={36} className="mx-auto mb-2 text-blue-500" />
                <p className="text-sm">Clique ou arraste PDFs aqui</p>
              </div>
            </label>

            <div className="mt-4 space-y-2 max-h-80 overflow-y-auto">
              {documents.map((doc) => (
                <div key={doc.id} className="flex items-center justify-between border border-slate-200 rounded p-2">
                  <div className="flex items-center gap-2 text-sm flex-1 min-w-0">
                    <FileText size={16} className="text-blue-500 flex-shrink-0" />
                    <div className="truncate">
                      <p className="font-medium truncate">{doc.file.name}</p>
                      <p className="text-xs text-slate-500">Label: {doc.label}</p>
                    </div>
                  </div>
                  <span className={`text-xs ${getStatusColor(doc.status)}`}>{doc.status}</span>
                  <button onClick={() => removeDocument(doc.id)} disabled={processing} className="ml-2 text-red-500">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>

            {documents.length > 0 && (
              <button
                onClick={processDocuments}
                disabled={processing}
                className="w-full mt-4 bg-blue-600 text-white rounded-lg py-2 text-sm hover:bg-blue-700 flex items-center justify-center gap-2 disabled:bg-slate-400"
              >
                {processing ? (
                  <>
                    <Clock size={16} className="animate-spin" /> Processando...
                  </>
                ) : (
                  <>
                    <Play size={16} /> Processar {documents.length} Documento{documents.length > 1 ? 's' : ''}
                  </>
                )}
              </button>
            )}

            {error && (
              <div className="mt-4 text-red-700 text-sm flex items-start gap-2">
                <AlertCircle size={16} /> {error}
              </div>
            )}
          </section>
        </div>

        {/* Resultados */}
        {results.length > 0 && (
          <section className="bg-white border border-slate-200 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Resultados</h2>
              <button
                onClick={exportResults}
                className="bg-green-600 text-white px-3 py-2 rounded text-sm hover:bg-green-700 flex items-center gap-2"
              >
                <Download size={16} /> Exportar JSON
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
              <div>
                <p className="text-slate-500">Documentos</p>
                <p className="text-xl font-semibold">{results.length}</p>
              </div>
              <div>
                <p className="text-slate-500">Tempo Médio</p>
                <p className="text-xl font-semibold">
                  {(results.reduce((acc, r) => acc + parseFloat(r.extractionTime), 0) / results.length).toFixed(2)}s
                </p>
              </div>
              <div>
                <p className="text-slate-500">Custo Total</p>
                <p className="text-xl font-semibold">
                  ${results.reduce((acc, r) => acc + parseFloat(r.cost), 0).toFixed(4)}
                </p>
              </div>
            </div>

            <div className="space-y-4 max-h-96 overflow-y-auto">
              {results.map((result) => (
                <div key={result.id} className="border-t border-slate-200 pt-3">
                  <h3 className="font-semibold">{result.fileName}</h3>
                  <p className="text-xs text-slate-600 mb-2">
                    {result.label} | {result.extractionTime}s | ${result.cost}
                  </p>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    {Object.entries(result.fields).map(([k, v]) => (
                      <div key={k} className="border border-slate-200 rounded p-2">
                        <div className="flex justify-between">
                          <span className="font-medium">{k}</span>
                          <span className={`text-xs ${getMethodColor(v.method)}`}>{v.method}</span>
                        </div>
                        <p className="text-slate-800">{v.value}</p>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 bg-slate-200 rounded-full h-1.5">
                            <div
                              className="bg-blue-500 h-1.5 rounded-full"
                              style={{ width: `${v.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-slate-500">{(v.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

export default DocumentExtractorUI;
