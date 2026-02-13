# GitHub 发布清单（论文复现包）

语言 / Language: [中文](GITHUB_RELEASE_CHECKLIST.zh-CN.md) | [English](GITHUB_RELEASE_CHECKLIST.md)

## 仓库元数据

- [x] `README.md` 含复现快速开始。
- [x] `LICENSE`（MIT）已提供。
- [x] `CITATION.cff` 已提供。
- [x] `CONTRIBUTING.md` 已提供。

## 复现入口

- [x] `reproduce.sh` 一键复现。
- [x] `Makefile` 分阶段流程。
- [x] `REPRODUCIBILITY.md` / `REPRODUCIBILITY.zh-CN.md` 复现说明。
- [x] `ARTIFACTS.md` / `ARTIFACTS.zh-CN.md` 产物映射。
- [x] `environment.yml` 环境定义已提供。

## 校验门禁

- [x] `scripts/preflight_release_check.py` 通过。
- [x] `scripts/academic_consistency_guard.py` 通过。
- [x] 关键回归测试通过。
- [x] CI 工作流：`.github/workflows/reproducibility.yml`。

## 可移植性

- [x] 仪表盘数据加载为仓库相对路径。
- [x] 应用入口中的本地绝对路径已移除。

## 发布前最后确认

- [x] `CITATION.cff` 已更新作者、身份与仓库地址。
- [ ] 确认哪些生成产物纳入仓库历史、哪些仅放 Release 附件。
- [ ] 编写 Release Notes（方法、复现范围、已知限制）。
