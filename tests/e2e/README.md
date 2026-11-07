# P0-1 Playwright Browser E2E

真实浏览器 E2E，覆盖：

- 客户在订单页创建退款 Case，进度显示“等待审核”；
- 审核台登录、选中 Case、批准，状态变为“已执行”；
- 二次批准返回“该 Case 已处理”，不产生第二次业务写入；
- 同一会话多个 Case：Case 选择器、分别批准/驳回；
- 客户进度更新为“已处理 / 未通过审核”；
- 跨用户读取 Case/进度 403，非法体验账号 401，越权选择订单 403；
- 审核队列空态、错误凭证、离线模式提示。

## 隔离方式

每个测试都会把 `app/`、`frontend/`、`policies/` 和运行数据复制到 `tmp` 目录，
在该目录启动独立 `uvicorn`（随机端口、独立 SQLite/checkpoint），并设置：

- `DEMO_MODE=1`
- `OPENAI_API_KEY=test-key`（offline_workflow，不调用线上 LLM）
- `REVIEW_API_TOKEN=e2e-review-token`（可用 `P0_E2E_REVIEW_TOKEN` 覆盖）

不会读写 V1 的进程、数据库或端口，也不会启动/重启线上服务。

## 依赖

Ubuntu 20.04 需要 Playwright `1.52.0`（更高版本已不再支持该发行版）：

```bash
python -m pip install "playwright==1.52.0"
python -m playwright install chromium
```

## 运行

```bash
tests/e2e/run.sh
# 或
RUN_E2E=1 python -m pytest tests/e2e -q
```

截图默认输出到 `reports/p0-e2e/`（可用 `P0_E2E_ARTIFACT_DIR` 覆盖）。

常规 `python -m pytest -q` 默认跳过本目录，保持 184+ 单元/集成测试的快速反馈。
