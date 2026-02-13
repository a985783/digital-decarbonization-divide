# 贡献指南

语言 / Language: [中文](CONTRIBUTING.zh-CN.md) | [English](CONTRIBUTING.md)

感谢你参与改进本论文复现包。

## 开发流程

1. 创建功能分支。
2. 每次提交保持单一逻辑变更。
3. 提交 PR 前运行本地检查：

```bash
make test
make verify
```

## 代码与复现规范

- 分析流程应保持确定性和可复现性。
- 禁止硬编码本地绝对路径。
- 非发布必需的生成产物不要提交到仓库历史。

## PR 检查清单

- [ ] `REPRODUCIBILITY.md` / `REPRODUCIBILITY.zh-CN.md` 中复现步骤仍可执行。
- [ ] `scripts/preflight_release_check.py` 通过。
- [ ] `scripts/academic_consistency_guard.py` 通过。
- [ ] 已补充/更新测试，或说明为何无需测试。
