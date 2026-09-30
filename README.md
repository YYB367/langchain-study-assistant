# PDF 学习资料问答助手

这是一个 Python 练习项目：先把 PDF 切分并保存到 Chroma，再通过自己编写的 FastAPI 接口提问。`POST /ask` 从已保存的检索库找相关段落，调用 DeepSeek 生成答案，并返回答案、PDF 页码、原文片段和耗时。`GET /health` 用于检查服务是否能响应。

项目中的 `src/01_chat_test.py` 至 `src/06_search_saved.py` 是逐步学习用的独立脚本；接口代码在 `src/api.py`，检索与回答函数在 `src/rag_service.py`。

## 准备环境（Windows PowerShell）

以下命令都在项目根目录运行。项目目前在 Python 3.12 环境下使用。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

复制示例配置文件为 `.env`，再在本机编辑 `.env`，填写自己的 `DEEPSEEK_API_KEY`。不要在终端打印密钥，也不要将 `.env` 上传到仓库。

```powershell
Copy-Item .env.example .env
```

把要检索的 PDF 放在 `data/六页资料文字版.pdf`。公开仓库默认不包含本机 PDF，克隆后需要自行放入资料。本项目的题库和预期页码针对这份六页 PDF；换资料后应重新建库并准备对应的评估题。首次使用中文向量模型时可能需要下载模型文件。

## 建立检索库：首次准备资料时运行

```powershell
.\.venv\Scripts\python.exe src\05_build_index.py
```

正常会看到“完成：已将 21 段保存到 ...”。脚本读取 PDF、切分文本，并把向量保存到项目的 `vector_db` 目录。已有检索库时跳过这一步；日常每次提问都不需要重新读取或切分 PDF。更换 PDF 后，旧库不会自动更新，需要重新建立与新资料对应的库。

## 启动接口与提问

在一个 PowerShell 窗口启动服务，并保持该窗口运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

在另一个 PowerShell 窗口检查服务：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常会看到 `status` 为 `ok`。接着提交一个问题：

```powershell
$body = @{ question = 'ATP是什么？' } | ConvertTo-Json -Compress
$response = Invoke-WebRequest -Method Post -Uri 'http://127.0.0.1:8000/ask' -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing
$data = [System.Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray()) | ConvertFrom-Json
$data.answer
$data.sources | Select-Object page, text
$data.elapsed_ms
```

正常会得到依据 PDF 的答案、第 1 页等来源页码、对应原文，以及毫秒数表示的耗时。中文使用 UTF-8 解码是为了避免部分 Windows PowerShell 环境显示乱码。`/ask` 会拒绝空问题；没有找到足够相关的片段时，返回说明文字和空的 `sources`；DeepSeek 请求失败时返回 HTTP 502。

## 评估

`evaluation.md` 保存 26 道题的预期要点、来源页码和人工判定。本机的 `evaluation_api_results.json` 保存初次批量调用的原始结果，默认不上传；表中标明了哪些题在修正检索后复测，二者可能不同。

先只检查 Chroma 是否检索到预期页码，不调用 DeepSeek：

```powershell
.\.venv\Scripts\python.exe src\run_evaluation.py --retrieval-only
```

在命令末尾加题号（例如 `--retrieval-only 6 10`）可以只检查指定题目。`run_evaluation.py` 不带参数时会调用本地 `/ask` 运行前三道**尚未判定且未保存原始结果**的题；`--all` 会运行所有这类待测题。当前 26 题已经判定，因此这些调用模式会跳过已有记录。

页码符合只说明检索到预期页。最终仍须检查答案要点、`sources.text` 是否支持答案，以及页码是否正确。这 26 题也用于调整检索，因此目前的通过结果不能保证新 PDF 或新问题同样通过。
