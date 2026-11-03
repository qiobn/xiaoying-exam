import React, { useEffect, useState } from 'react';
import { Check, LoaderCircle, Save, TestTube2 } from 'lucide-react';

const api = async (path, options = {}) => {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.text()) || response.statusText);
  return response.json();
};

const initial = {
  mode: 'mock', model: 'gpt-4o-mini', base_url: '', api_key: '', api_key_configured: false,
  llm_timeout_seconds: 45, query_rewrite_enabled: true, local_models_only: true,
  embedding_backend: 'none', embedding_model: 'BAAI/bge-small-zh-v1.5', embedding_device: 'cpu', embedding_batch_size: 16, model_cache_dir: 'runtime/models', embedding_api_url: '', embedding_api_key: '', embedding_api_key_configured: false,
  rerank_enabled: true, rerank_backend: 'local', rerank_model: 'BAAI/bge-reranker-base', rerank_device: 'cpu', rerank_top_k: 8, rerank_api_url: '', rerank_api_key: '', rerank_api_key_configured: false,
  mineru_command: '', mineru_server_url: '', mineru_method: 'auto',
  tesseract_command: 'tesseract', tesseract_lang: 'chi_sim+eng',
  chunk_strategy: 'auto', chunk_llm_model: '', chunk_llm_min_chars: 1600, chunk_llm_max_chars: 12000, chunk_llm_timeout_seconds: 60,
  parse_pdf_tables: true, parse_embedded_images: true,
  vision_enabled: false, vision_model: '', vision_api_url: '', vision_api_key: '', vision_api_key_configured: false,
  vision_timeout_seconds: 60, max_embedded_images: 20,
};

export default function SettingsPanel({ capabilities, onSaved }) {
  const [form, setForm] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState('');
  const [connection, setConnection] = useState(null);
  const [clearKey, setClearKey] = useState(false);
  const [clearRerankKey, setClearRerankKey] = useState(false);
  const [clearVisionKey, setClearVisionKey] = useState(false);

  useEffect(() => {
    api('/api/settings').then(value => setForm(current => ({ ...current, ...value, api_key: '', embedding_api_key: '', rerank_api_key: '', vision_api_key: '' })))
      .catch(error => setMessage(error.message));
  }, []);

  const change = event => setForm(current => ({ ...current, [event.target.name]: event.target.value }));
  const save = async event => {
    event.preventDefault();
    setBusy(true); setMessage('');
    try {
      const payload = { ...form, clear_api_key: clearKey, clear_rerank_api_key: clearRerankKey, clear_vision_api_key: clearVisionKey };
      if (!form.api_key) delete payload.api_key;
      if (!form.rerank_api_key) delete payload.rerank_api_key;
      if (!form.embedding_api_key) delete payload.embedding_api_key;
      if (!form.vision_api_key) delete payload.vision_api_key;
      const saved = await api('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      setForm(current => ({ ...current, ...saved, api_key: '', embedding_api_key: '', rerank_api_key: '', vision_api_key: '' }));
      setClearKey(false); setClearRerankKey(false); setClearVisionKey(false); setMessage('配置已保存并立即应用；更换分块、Embedding 或解析策略后请重建知识库索引'); setConnection(null);
      onSaved(await api('/api/capabilities'));
    } catch (error) { setMessage(`保存失败：${error.message}`); }
    finally { setBusy(false); }
  };
  const testModel = async () => {
    setChecking(true); setConnection(null);
    try {
      const payload = { ...form };
      if (!form.api_key) delete payload.api_key;
      if (!form.rerank_api_key) delete payload.rerank_api_key;
      if (!form.vision_api_key) delete payload.vision_api_key;
      const result = await api('/api/settings/test-model', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      setConnection(result);
    } catch (error) { setConnection({ ok: false, message: error.message }); }
    finally { setChecking(false); }
  };
  const parsers = capabilities?.parsers;

  return <>
    <div className="page-title"><div><div className="eyebrow">SETTINGS</div><h1>模型与解析设置</h1><p>配置保存至 settings/.env（已加入 Git 忽略）；密钥不回显。</p></div></div>
    <form className="panel settings-card settings-form" onSubmit={save}>
      <h2>问答模型</h2>
      <label className="form-field"><span>运行模式</span><select name="mode" value={form.mode} onChange={change}><option value="mock">离线 Mock（流程测试）</option><option value="real">真实 OpenAI-compatible 模型</option></select></label>
      <label className="form-field"><span>模型名称</span><input name="model" value={form.model} onChange={change} placeholder="例如 gpt-4o-mini" /></label>
      <label className="form-field"><span>API Base URL</span><input name="base_url" value={form.base_url} onChange={change} placeholder="留空使用供应商默认地址" /></label>
      <label className="form-field"><span>API Key {form.api_key_configured && <small>已配置，留空保留当前密钥</small>}</span><input name="api_key" type="password" autoComplete="new-password" value={form.api_key} onChange={change} placeholder={form.api_key_configured ? '已配置（输入新值可替换）' : '输入服务密钥'} /></label>
      <label className="form-field"><span>模型超时（秒）</span><input type="number" min="5" max="180" name="llm_timeout_seconds" value={form.llm_timeout_seconds} onChange={change} /></label>
      <label className="check-field"><input type="checkbox" checked={form.query_rewrite_enabled} onChange={event => setForm(current => ({ ...current, query_rewrite_enabled: event.target.checked }))} /> 启用上下文 query 改写</label>
      <label className="check-field"><input type="checkbox" checked={form.local_models_only} onChange={event => setForm(current => ({ ...current, local_models_only: event.target.checked }))} /> 本地模型仅使用缓存（离线优先）</label>
      {form.api_key_configured && <label className="check-field"><input type="checkbox" checked={clearKey} onChange={event => setClearKey(event.target.checked)} /> 清除已保存的 API Key</label>}
      <div className="settings-actions"><button className="secondary" type="button" onClick={testModel} disabled={checking}>{checking ? <LoaderCircle className="spin" size={15}/> : <TestTube2 size={15}/>}测试模型连接</button><button className="primary" type="submit" disabled={busy}>{busy ? <LoaderCircle className="spin" size={15}/> : <Save size={15}/>}保存配置</button></div>
      {connection && <div className={`connection-result ${connection.ok ? 'ok' : 'error'}`}>{connection.ok ? <Check size={15}/> : null}{connection.message || `${connection.model} 连接成功 · ${connection.duration_ms} ms`}{connection.preview ? ` · ${connection.preview}` : ''}</div>}
      {message && <div className="settings-message">{message}</div>}
      <h2>Embedding 与混合检索</h2>
      <div className="setting-row"><div><b>Embedding / 向量索引</b><small>默认推荐 BAAI/bge-small-zh-v1.5；本地首次构建会下载权重，依赖或模型不可用时退回 BM25</small></div><span className="tag">{capabilities?.retrieval?.embedding?.dependency_installed ? '本地依赖可用' : '本地依赖未安装'}</span></div>
      <label className="form-field"><span>Embedding 后端</span><select name="embedding_backend" value={form.embedding_backend} onChange={change}><option value="none">关闭向量检索（BM25）</option><option value="local">本地 Sentence Transformers</option><option value="openai_compatible">OpenAI-compatible Embeddings</option></select></label>
      <label className="form-field"><span>Embedding 模型</span><input name="embedding_model" value={form.embedding_model} onChange={change} placeholder="BAAI/bge-small-zh-v1.5" /></label>
      <label className="form-field"><span>设备</span><select name="embedding_device" value={form.embedding_device} onChange={change}><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">Apple MPS</option></select></label>
      <label className="form-field"><span>批处理大小</span><input type="number" min="1" max="256" name="embedding_batch_size" value={form.embedding_batch_size} onChange={change} /></label>
      <label className="form-field"><span>模型缓存目录</span><input name="model_cache_dir" value={form.model_cache_dir} onChange={change} /></label>
      {form.embedding_backend === 'openai_compatible' && <><label className="form-field"><span>Embedding API URL</span><input name="embedding_api_url" value={form.embedding_api_url} onChange={change} placeholder="可与问答模型不同" /></label><label className="form-field"><span>Embedding API Key {form.embedding_api_key_configured && <small>已配置，留空保留</small>}</span><input name="embedding_api_key" type="password" autoComplete="new-password" value={form.embedding_api_key} onChange={change} /></label></>}
      <h2>Rerank 重排</h2>
      <label className="check-field"><input type="checkbox" checked={form.rerank_enabled} onChange={event => setForm(current => ({ ...current, rerank_enabled: event.target.checked }))} /> 启用候选重排</label>
      <label className="form-field"><span>Rerank 后端</span><select name="rerank_backend" value={form.rerank_backend} onChange={change}><option value="local">本地 CrossEncoder</option><option value="dashscope">DashScope</option><option value="http">通用 HTTP Rerank</option><option value="none">关闭</option></select></label>
      <label className="form-field"><span>Rerank 模型</span><input name="rerank_model" value={form.rerank_model} onChange={change} placeholder="BAAI/bge-reranker-base" /></label>
      <label className="form-field"><span>设备 / 候选数</span><div className="inline-fields"><select name="rerank_device" value={form.rerank_device} onChange={change}><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">Apple MPS</option></select><input aria-label="Rerank 候选数" type="number" min="1" max="100" name="rerank_top_k" value={form.rerank_top_k} onChange={change} /></div></label>
      {form.rerank_backend === 'http' && <><label className="form-field"><span>Rerank API URL</span><input name="rerank_api_url" value={form.rerank_api_url} onChange={change} /></label><label className="form-field"><span>Rerank API Key {form.rerank_api_key_configured && <small>已配置，留空保留</small>}</span><input type="password" autoComplete="new-password" name="rerank_api_key" value={form.rerank_api_key} onChange={change} /></label>{form.rerank_api_key_configured && <label className="check-field"><input type="checkbox" checked={clearRerankKey} onChange={event => setClearRerankKey(event.target.checked)} /> 清除 Rerank API Key</label>}</>}
      <div className="setting-row"><div><b>Rerank 能力</b><small>{capabilities?.retrieval?.rerank?.dependency_installed ? '本地 sentence-transformers 依赖已安装' : '本地重排需要安装可选依赖；远程后端独立配置'}</small></div><span className="tag">{capabilities?.retrieval?.rerank?.available ? (capabilities?.retrieval?.rerank?.configured ? '可用' : '关闭') : '待安装依赖'}</span></div>
      <h2>智能分块与图表理解</h2>
      <label className="form-field"><span>文本分块策略</span><select name="chunk_strategy" value={form.chunk_strategy} onChange={change}><option value="auto">自动（普通长文使用 LLM）</option><option value="structured">标题与段落结构感知</option><option value="llm">LLM 语义分块（失败自动降级）</option></select></label>
      <label className="form-field"><span>分块模型</span><input name="chunk_llm_model" value={form.chunk_llm_model} onChange={change} placeholder="留空复用问答模型" /></label>
      <label className="form-field"><span>LLM 最小触发字符 / 单窗口上限</span><div className="inline-fields"><input aria-label="LLM 最小触发字符" type="number" min="300" name="chunk_llm_min_chars" value={form.chunk_llm_min_chars} onChange={change} /><input aria-label="LLM 单窗口上限" type="number" min="1000" name="chunk_llm_max_chars" value={form.chunk_llm_max_chars} onChange={change} /></div></label>
      <label className="form-field"><span>分块模型超时（秒）</span><input type="number" min="5" max="300" name="chunk_llm_timeout_seconds" value={form.chunk_llm_timeout_seconds} onChange={change} /></label>
      <label className="check-field"><input type="checkbox" checked={form.parse_pdf_tables} onChange={event => setForm(current => ({ ...current, parse_pdf_tables: event.target.checked }))} /> 提取 PDF 表格并转换为 Markdown</label>
      <label className="check-field"><input type="checkbox" checked={form.parse_embedded_images} onChange={event => setForm(current => ({ ...current, parse_embedded_images: event.target.checked }))} /> 解析 DOCX/PDF 内嵌图片</label>
      <label className="form-field"><span>单文档最多图片数</span><input type="number" min="0" max="200" name="max_embedded_images" value={form.max_embedded_images} onChange={change} /></label>
      <label className="check-field"><input type="checkbox" checked={form.vision_enabled} onChange={event => setForm(current => ({ ...current, vision_enabled: event.target.checked }))} /> 启用视觉模型描述图表与截图</label>
      {form.vision_enabled && <><label className="form-field"><span>视觉模型</span><input name="vision_model" value={form.vision_model} onChange={change} placeholder="留空复用问答模型" /></label><label className="form-field"><span>视觉 API Base URL</span><input name="vision_api_url" value={form.vision_api_url} onChange={change} placeholder="留空复用问答模型地址" /></label><label className="form-field"><span>视觉 API Key {form.vision_api_key_configured && <small>已配置，留空保留</small>}</span><input name="vision_api_key" type="password" autoComplete="new-password" value={form.vision_api_key} onChange={change} /></label><label className="form-field"><span>视觉超时（秒）</span><input type="number" min="5" max="300" name="vision_timeout_seconds" value={form.vision_timeout_seconds} onChange={change} /></label>{form.vision_api_key_configured && <label className="check-field"><input type="checkbox" checked={clearVisionKey} onChange={event => setClearVisionKey(event.target.checked)} /> 清除视觉 API Key</label>}</>}
      <h2>文档解析</h2>
      <div className="setting-row"><div><b>MinerU PDF 解析</b><small>命令和 HTTP 服务均可在下方配置；未配置时使用 PyMuPDF</small></div><span className="tag">{parsers?.mineru?.available ? '可用' : '未配置'}</span></div>
      <label className="form-field"><span>MinerU 命令</span><input name="mineru_command" value={form.mineru_command} onChange={change} placeholder="mineru 或本机完整路径" /></label>
      <label className="form-field"><span>MinerU 服务地址</span><input name="mineru_server_url" value={form.mineru_server_url} onChange={change} placeholder="可选 HTTP VLM 服务地址" /></label>
      <label className="form-field"><span>MinerU 方法</span><input name="mineru_method" value={form.mineru_method} onChange={change} /></label>
      <div className="setting-row"><div><b>Tesseract OCR</b><small>图片与扫描 PDF 识别 · 语言 {form.tesseract_lang}</small></div><span className="tag">{parsers?.ocr?.available ? '可用' : '未安装'}</span></div>
      <label className="form-field"><span>Tesseract 命令</span><input name="tesseract_command" value={form.tesseract_command} onChange={change} /></label>
      <label className="form-field"><span>OCR 语言</span><input name="tesseract_lang" value={form.tesseract_lang} onChange={change} placeholder="chi_sim+eng" /></label>
      <div className="runtime-summary"><b>当前运行时</b><span>{capabilities?.agent?.mode || '读取中'} · {capabilities?.agent?.model || '—'}</span><span>{capabilities?.retrieval?.method || '—'} · Embedding {capabilities?.retrieval?.embedding_model || 'none'}</span></div>
    </form>
  </>;
}
