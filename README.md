# franka-ik —— Franka Emika Panda 解析逆运动学

**中文** | [English](README.en.md)

这是 2020 年底为 Franka Emika Panda 推导的解析（闭式）逆运动学。Panda 本身不是 S-R-S 构型
机械臂，所以先把手臂**等效**成一台 S-R-S 机械臂，再用
[Shimizu 等 (2008)](https://doi.org/10.1109/TRO.2008.2003266) 的闭式解求解。冗余自由度用
**关节 7** 参数化：给定目标位姿和想要的 q₇，返回所有满足条件的构型。

Panda 的肩部偏置 0.0825 m、腕部偏置 0.088 m，因此这个化归并非平凡。`docs/method.md`
逐式对照手写推导（仓库根目录的 `franka解析反解方法.pdf`，`STEP1`–`STEP4`）。

```python
import numpy as np
import franka_ik as fk

q = np.array([0.3, -0.4, 0.2, -1.2, 0.1, 1.0, 0.5])   # 任意一个合法构型
pose = fk.fk_flange(q)

solutions = fk.solve(pose, q7=q[6], within_limits_only=True)
print(solutions[0].describe())
# q4+ phi+ flip    q = [ 17.189, -22.918,  11.459, -68.755,   5.730,  57.296,  28.648] deg   pose error 3.75e-16   in limits
```

## 主要结论

当年发布的实现认为有**四个**分支（其 readme 是这么写的，也确实提供了两个文件、各两个入口）。
实际上有**八个**。`STEP2` 的肘部方程是关于 `tan(θ₄/2)` 的一元二次方程，而原代码只取了 `+` 根：

```python
tan_half_q4 = (a1 + ca.sqrt(a1**2 - 4*a0*a2)) / (2*a2)     # original/ik_ca.py
```

两个根都保留后分支数翻倍，效果是可测量的：

| 指标（300 个关节限位内随机构型，`seed 0`） | 原四分支 | 本库八分支 |
|---|---|---|
| 能反解回原构型 | 265/300 = **88.3 %** | 300/300 = **100 %** |
| 返回解的位姿残差 | ≤ 2.0 × 10⁻¹⁴ | ≤ 2.0 × 10⁻¹⁴ |
| 单个位姿在限位内的不同解个数 | — | 平均 **3.15**，范围 1–8 |
| 全部候选分支耗时（纯 NumPy） | — | 约 1.3 ms/位姿 |

关键区别在于：原有的四个分支**都是对的**，只是不全。漏掉分支的解算器依然会返回四个精确的
位姿，唯一能发现问题的检验是"能不能把出发点原样还给我"——这正是 `analysis.coverage_study`
所做的测量，细节见 `docs/branch_analysis.md`。300 个目标中丢失的那 35 个，恰好就是肘部落在
二次方程第二个根上的构型。

这也不只是"冗余自由度选得不够好"的问题：在 150 个随机构型上，原四分支给出的限位内解比八分支
少 18.6%；其中甚至有一个位姿，原四分支一个解都给不出来（把可达位姿判成了不可达），而八分支
能找到四个解。

因此，保留两个根是对原推导的**补充**，而不是对它的推翻。

## 与原实现的交叉验证

`original/` 里是 2020 年发布时的文件，一字未改。本库与之逐项对照：

| 检验项 | 结果 |
|---|---|
| NumPy 模型与 CasADi 模型（`fk_flange`、`fk_tool`），300 个随机构型 | 3.3 × 10⁻¹⁶ |
| 本库与原解算器逐分支对照（覆盖整个关节 7 区间） | 1200/1200 一致，最大偏差 **1.30 × 10⁻¹³ rad** |
| 返回解的位姿残差 | 中位数 4.4 × 10⁻¹⁶，最大 2.0 × 10⁻¹⁴ |
| 肩部翻转对称 `(q₁,q₂,q₃) → (q₁+π, −q₂, q₃+π)` | 138/138 构型成立 |
| 腕部翻转对称 `(q₅+π, −q₆, q₇+π)` | 0/138 —— Panda 不是真正的 S-R-S 构型 |
| 用独立优化器（CasADi + IPOPT）反查是否有漏掉的分支 | 15 个位姿共 56 个解，**0 个反例**，最远距离 0.085° |

原代码中还有三处缺陷被记录下来而不是悄悄改掉，因为它们都特别费时间：`limit_joints` 遇到
`nan` 会死循环、`Panda.fk` 返回符号对象无法转成数值、关节 4 和关节 6 的限位窗口比一整圈还宽。
第四处出在腕部符号判据上，这一处是直接**修好**的：该判据在 `q₇ = ±90°` 处退化并给出错误位姿，
而本库改为在腕部两个候选构型中挑选真正能到达目标位姿的那个。详见 `docs/limitations.md`。

## 安装

环境用 [uv](https://docs.astral.sh/uv/) 管理：

```bash
git clone git@github.com:yangliangzhu/franka-emika-inverse-kinematics.git
cd franka-emika-inverse-kinematics
uv sync                 # 建 .venv 并装好 dev 组（pytest / casadi / ruff）
uv run pytest -q        # 131 个测试
uv run ruff check .
```

`uv.lock` 已提交，所以 `uv sync` 会复现测试时的确切版本；`uv run <命令>` 不必手动激活
虚拟环境。不想用 uv 的话：

```bash
pip install -e .                    # numpy + matplotlib，求解器是纯 NumPy
pip install -r requirements.txt     # 额外装上 CasADi，只有跑 original/ 时才需要
```

要求 Python ≥ 3.10（`viz` 依赖 `roboticstoolbox`，它要求 3.10；库本身仍然只需要 NumPy）。`franka_ik` 本身只依赖 NumPy；matplotlib 用于画图。**CasADi 是可选的** ——
求解器不用它（纯 NumPy，8 个分支约 1 ms），但 `original/` 里的原实现用，所以逐支对照的测试需要
它。没装 CasADi 时那部分测试会跳过（107 passed, 18 skipped）而不是失败。

## 使用

```python
import numpy as np
import franka_ik as fk

pose = fk.fk_flange(np.array([0.3, -0.4, 0.2, -1.2, 0.1, 1.0, 0.5]))   # 4x4 位姿由你提供
q7   = 0.5                                # 冗余参数，单位弧度

fk.solve(pose, q7, within_limits_only=True)      # 已校验、在限位内、已去重
fk.branch_solutions(pose, q7)                    # 全部分支，不做筛选
fk.solve_closest(pose, q7, reference=previous)   # 离参考构型最近的一支，用于跟踪
```

`solve` 会用正运动学校验每个候选解，残差超过 `tolerance`（默认 `1e-9`）的直接丢弃，所以返回
的一定是对的；但它不保证一定有解，而且有没有解取决于你给的 `q₇`。接进控制器之前，有三点需要
知道：

* **`q₇` 要跟踪，不能固定。** 固定一个 `q₇` 时，可达位姿里只有大约三分之一能解出来；而沿用上
  一条指令的 `q₇`，同一批样本是 200/200。
* **求解器针对的是法兰（flange）坐标系**，不是工具坐标系。换算关系是
  `T_flange = T_tool @ rot_z(+π/4) @ trans_z(-0.1034)`；漏掉 0.1034 m 这一项会得到"看起来正常、
  其实是错的"结果。
* **`q₇ = ±90°` 曾经是最脆弱的地方。** 原实现的腕部符号判据正比于 `cos²(q₇)`，在那里退化为零，
  会返回错误位姿（在整个关节 7 区间上扫过 7896 次调用，16 次落在错误位姿上，全部出现在 `±90°`）。
  本库没有这个问题：它会分别算出腕部的两个候选构型，保留真正能到达目标位姿的那个；在 `±90°`
  处 100/100 都能反解回原构型。

这三点背后的测量都在 `docs/limitations.md`。

## 目录结构

| 路径 | 说明 |
|---|---|
| `franka_ik/` | 库本体：`model`（改进 DH 正解、雅可比）、`geometry`（S-R-S 化归）、`solver`（八分支解析反解）、`analysis`（解集研究），以及负责画图与演示页面的 `viz`、`report` |
| `original/` | 2020 年的实现，原样保留：`panda.py`、`ik_ca.py`、`ik_ca2.py`、`test.ipynb` |
| `docs/` | `method.md`、`branch_analysis.md`、`limitations.md`、`api.md`，以及作者当年的 `original_notes_zh.md` |
| `franka解析反解方法.pdf` | 手写推导，`STEP1`–`STEP4` |
| `tests/` | 测试套件，包含与 `original/` 的逐分支对照 |
| `examples/` | 11 个例子：`01`–`06` 是 matplotlib 的推导走读（模型、化归、八个分支、关节限位、覆盖率、跟踪），`07`–`11` 是 Swift 交互式 3D 演示 |
| `demos/` | 自带样式的交互式 HTML 页面，浏览器直接打开 `demos/index.html` 即可，不需要服务器或构建 |
| `third_party/` | 内置的 Panda 描述：`franka_description` 的 Apache-2.0 子集（约 11 MB，8 个视觉网格、30 个碰撞体），`--model mesh` 画的就是它，出处见 `third_party/README.md` |
| `scripts/` | 研究脚本（`study_branches.py`、`study_wrist_offset_ik.py`）、演示页自检 `check_demo.js`，以及用 Playwright 驱动可视化做端到端自检的 `browser_drive.py` |

## 文档

| 文档 | 内容 |
|---|---|
| [docs/method.md](docs/method.md) | 推导全过程，按 `STEP1`–`STEP4` 逐式展开，并标注 S-R-S 论文的公式号 |
| [docs/branch_analysis.md](docs/branch_analysis.md) | 四个分支还是八个：测量方法，以及为什么最直观的检验看不出问题 |
| [docs/browser_debugging.md](docs/browser_debugging.md) | 窗口空白、动画不动：踩过的 Swift API 坑，以及怎么用 Playwright 直接看页面 |
| [docs/limitations.md](docs/limitations.md) | 关节限位、`q₇ = ±90°`、并不存在的对称性、原代码的缺陷 |
| [docs/provenance.md](docs/provenance.md) | 来龙去脉的实测版：时间线、最接近的已发表工作、以及它对同一个第二根的取舍 |
| [docs/api.md](docs/api.md) | 全部导出符号，含签名与示例 |
| [docs/original_notes_zh.md](docs/original_notes_zh.md) | 作者 2020 年写下的原始说明 |
| [AGENTS.md](AGENTS.md) | 参与开发指南：环境、测试、代码风格 |

## 让它动起来

上面有几条结论，看着机械臂动一遍比读数字容易信：八个分支、可达外壳、关节 2 过零。
`examples/07`–`11` 是基于 Swift 的交互式 3D 演示，由 `solve` 用的同一套闭式解驱动：

```bash
uv sync --extra viz
python3 examples/07_swift_branches.py --pose second_root   # 同一个位姿的全部解，按肘根着色
python3 examples/08_swift_workspace.py                      # 可达外壳，配合 --scan
python3 examples/10_swift_singularities.py                  # q2 = 0：没有既精确又在限位内的解
python3 examples/11_swift_tracking.py                       # 用 solve_closest 跟踪一条直线
```

Swift 窗口是**可交互的**，而且每个控件都对应一条测量：`07` 用关节 7 滑块重解并重建整扇解，配播放按钮与相机切换；`08` 用**肘部**（关节 4，唯一能改变 `‖x_sw‖` 的单关节）滑块把腕部从 0.20685 m 拖到 0.71935 m——正好是闭式外半径；`09` 扫关节 7 并配“每个肘根一个 / 全部解”单选；`10` 用关节 2 偏移滑块（±0.01°，步长 1e-5）夹住只有 1e-4° 宽的失败窗口，另有四个实测偏移的预设；`11` 可擦洗、可播放一整条笛卡尔直线，并按“步/秒”调速。每个窗口都带一行实时读数。`demos/branches.html` 不用 Swift 也能动：关节 7 滑块 + 播放按钮，扫掠数据由 `franka_ik.report` 预先算好内嵌，颜色**蓝色＝公布代码保留的那个肘根，橙色＝它丢掉的那个**——拖到生成构型就能看见它落在橙色一侧。

需要浏览器（或者用 `--headless`：照样跑完并打印全部数字，**但不画**——只看到文字是这个参数造成的，
不是缺 URDF）。

`--model` 决定画成什么样，三种都能离线跑：

* **`--model mesh`（默认）** 画 Panda **自己的视觉网格**，也就是真机。描述文件内置在
  `third_party/`（Apache-2.0，约 11 MB，8 个 link），首次使用时从 xacro 展开；网格位置由
  `franka_ik.model.forward_kinematics` 给出，所以同样不可能和求解器脱节；
* `--model collision` 画 `rtb-data` 自带描述里的碰撞体：30 个圆柱和球，除已安装的包之外不需要
  任何文件；
* `--model skeleton` 是火柴人，也是唯一能按肘根分别上色的模式——07 和 09 靠它区分两个分支半区。

```bash
uv run --extra viz python examples/07_swift_branches.py --pose second_root --browser auto
uv run --extra viz python examples/07_swift_branches.py --pose second_root --model collision
```

`FRANKA_IK_URDF` 可以换成你自己的 URDF，但会**先和 `fk_tool` 校验再画**——`franka_description`
里同时有 Panda 和 FR3，两者腕部差 57 mm，拿 Panda 的关节角去驱动 FR3 会画出一个看着合理、
其实错误的机械臂。出处见 [third_party/README.md](third_party/README.md)；11 个例子都在
[examples/README.md](examples/README.md)。

## 来龙去脉

这是一套自行推导的方法，**2021 年 2 月**首次推到 GitHub，当时作者刚接触 Franka 机械臂，
在网上找不到 Panda 的解析反解。思路是把腕部偏置旋转进一段加长的连杆，把机械臂化归为
一族**等效** S-R-S 构型，再用 KUKA（S-R-S）闭式解求解，冗余量取关节 7。

但它**并不新颖**，仓库不该装作新颖：He 与 Liu 在 ICRA 2022 发表了同样的化归、同样的冗余
参数、同样的八个分支（[IEEE Xplore 9646185](https://ieeexplore.ieee.org/abstract/document/9646185)）。
他们的预印本比这里的首次提交晚约八个月，但"没人看的仓库"不算科学优先权。本仓库相对已发表
工作真正多做的一件事，是**测量自己的完备性**——与原代码逐分支比对、用独立的 IPOPT 枚举
反向验证、统计分支数——并把自己的错误记下来。时间线、并排数据和"哪些部分真值钱"的排序
都在 [docs/provenance.md](docs/provenance.md)。

2020/2023 年的代码原封不动保存在 `original/` 里，`docs/original_notes_zh.md` 是作者当年的
说明，推导本身则是仓库根目录那份手写 PDF。

使用的话，请引用方法所依赖的已发表出处：

> M. Shimizu, H. Kakuya, W.-K. Yoon, K. Kitagaki, K. Kosuge, "Analytical Inverse Kinematic
> Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to
> Redundancy Resolution", *IEEE Transactions on Robotics*, 24(5):1131–1142, 2008.
> [doi:10.1109/TRO.2008.2003266](https://doi.org/10.1109/TRO.2008.2003266)

> Y. He, S. Liu, "Analytical Inverse Kinematics for Franka Emika Panda — a Geometrical Solver for
> 7-DOF Manipulators with Unconventional Design", *IEEE International Conference on Robotics and
> Automation (ICRA)*, 2022.

## 许可证

MIT，见 [LICENSE](LICENSE)。
