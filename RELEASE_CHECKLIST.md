# 发布前核验清单

当前项目仓库：<https://github.com/clmjan/RAG->。发布简历前，仍应在干净网络环境逐项执行下面的检查。

## 仓库与 README

```powershell
$REPO_URL = "https://github.com/clmjan/RAG-"
git remote -v
Invoke-WebRequest -Method Head -Uri $REPO_URL -MaximumRedirection 5
Invoke-WebRequest -Method Head -Uri "$REPO_URL/blob/main/README.md" -MaximumRedirection 5
```

应确认：仓库为公开状态、默认分支确实包含 `README.md`、README 中的相对路径和命令与仓库内容一致。
如果使用 Gitee，把 URL 替换为 `https://gitee.com/<owner>/<repo>`，不要只填写个人主页。

## 密钥扫描

```powershell
rg -n -i --hidden --glob '!.git/**' --glob '!frontend/node_modules/**' `
  'api[_-]?key|secret|token|authorization: bearer|sk-[A-Za-z0-9]' .
```

允许出现的是 `.env.example` 中的空值或 `replace-me` 占位符，以及测试中的假值；真实 API Key
只能放在本地 `.env` 或 CI Secret，不能提交 `.env`、日志、截图或数据库导出文件。

## 可运行性

```powershell
pytest -q
python -m compileall -q app tests
python -m app.seed_demo
python -m app.evaluate --compare
```

Docker 主路径：

```powershell
Copy-Item .env.example .env
docker compose config
docker compose up --build -d
Invoke-WebRequest http://127.0.0.1:8000/ready
docker compose down
```

将 `data/evaluation_results.md` 中的结果替换为发布前最后一次实际运行的结果，并记录运行日期、
数据集 SHA-256、是否配置外部 Embedding/LLM Provider。未实际运行 Docker 时，不要在简历中写“Docker
已验证”，只能写“提供 Docker 启动配置”。
