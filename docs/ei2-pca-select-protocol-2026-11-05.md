# EI-PS1：自动筛选具体PCA方向

用户授权验证现行版本3的EI-2-3-3：-3-1 L1、-3-2 Elastic Net、-3-3 ARD。原14F始终提供，只筛每层32个新增PCA方向，共448个候选，可跳选，每层0..32。筛完丢弃选择器临时头，重训原L2逻辑回归；不将选择器的分类头变化混入最终头。

## 固定条件

沿用EI-V1原Z/F、EI-PCA2训练侧PCA32、原末位和0基16～29层、train1600/dev600/旧safe-cal400/历史C1000/RV1082/OR-cal400及test1500。源封印、父样本标签角色、缓存/基底SHA先检查。标准化只fit训练，F除sqrt14、PCA坐标统一除sqrt56，按固定层/方向顺序压缩坐标并补零512。全部已观察，只是历史诊断。

选择器只用1600训练标签；dev600选择预定参数和最终C，校准及测试不参与。最终L2头liblinear、C=[.1,1,10]、max_iter5000，种子20261104/05/06；按三种子dev AUROC均值选参数/C，平局先少方向再列表顺序。不合并dev重训，主报告首种子。

## 三个选择器

L1和Elastic Net均为二分类logistic选择器，本项目用非负变量拆分wPCA=u−v、L-BFGS-B求解，避免给原F施加L1。临时目标为平均BCE＋||w||²/(2*10*N)＋lambda*[rho*||wPCA||1+(1−rho)*||wPCA||²/2]；截距不惩罚。L1的rho=1，Elastic Net的rho=.5，lambda各[.0001,.001,.01]。全部float64，零初始化，maxiter5000、maxls50、ftol1e-14、gtol1e-7；原目标KKT无穷范数≤2e-6才接受。仅新增方向按|w|>1e-8保留，原F不受删除。

ARD采用适用于二分类的logistic-Laplace近似：给PCA系数独立高斯精度alpha，给F固定精度.1、截距1e-6。每轮求带高斯先验的logistic MAP，Sigma=(X' diag(p(1−p)) X+A)^−1，gamma=1−alpha*diag(Sigma)，alpha目标gamma/(mu²+1e-24)。初始PCA alpha1，目标clip[1e-6,1e12]，log空间.5阻尼；第5轮后alpha≥1e10数值剪枝。最多100外轮，连续3轮无剪枝且最大log-alpha变化<1e-3可停止；最后再求一次MAP/协方差。Newton-Cholesky＋Armijo用于每次MAP，最多60步，梯度≤2e-5必须满足。外轮未达固定点要如实标记“迭代上限下的近似”，不声称完全收敛。

ARD只拟合一次分类后验，随后用alpha阈值[1e2,1e3,1e4]产生三个mask，已数值剪枝方向不回收。该实现借鉴Tipping2001分类部分，不是作者RVM程序或核模型的精确复现，不使用ARDRegression充当分类。

L1/Elastic Net正则与稀疏特征选择参考[官方说明](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)；分类Laplace推断参考[Tipping2001第3节](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf)。

## 匹配对照、稳定性和预算

每方法3个mask候选×3C×3种子，共27最终头，三方法81拟合。选择器确定性，不伪称三seed是三次独立选择。所选mask各层数量确定后，保留同样每层数量的前缀作为匹配对照，用所选相同C和三个seed各拟合一次，再9头；不另调前缀C。该对照条件于候选所选数量/参数，不是前缀超参优化冠军。

主比较6项：每候选−全部32方向，以及每候选−自身同数量前缀。每个历史批次共享10000次标签分层配对bootstrap，每项双侧99.166667%（尾.05/12）。固定8/16、B-F与前轮CV12描述性参照；不按测试改mask/阈值或升级比较。各批分别分析，不声称跨批联合确认。

在冻结选择与最终头后，训练侧另做5次分层bootstrap（每类800、有放回，种子20261106）检查所选超参数的mask稳定性。原PCA基底与标准化固定，只重新采样训练行并重训选择器，三方法共15次，ARD沿用所选alpha阈值。不看测试选稳定性赢家。报告原mask和重采样mask保留数、Jaccard；数量越多本就容易高重合，不能仅凭Jaccard宣称更稳定。此项是固定预处理下条件稳定性，不覆盖PCA/标准化变动。

共90最终LR（81搜索＋9匹配前缀）、初始7选择器（3L1＋3EN＋1ARD）、15稳定性选择器＝22次选择器拟合。不同求解器/迭代成本不等，必须报告KKT、ARD内外收敛与全部失败。额外选择预算相对固定32不同，不称纯信息因果证明。

旧safe400第381顺序统计量、严格大于告警；另报OR-safe400重校准。四项指标为AUROC、标准化pAUC(FPR≤.1)、检出率、误报率/条数；三种子全留，逐层具体方向编号和数量、是否前缀、成本/容量/稳定性均报告。

代码/协议先提交，新独立工作树results/ei2-pca-select，不改旧冻结文件。独立验证重载所选头、参考头与选择器状态，检查训练标准化、L1/EN KKT、ARD最终MAP梯度/协方差/alpha mask、搜索选择规则、同数量/同C前缀、全部分数和两套阈值、全部18主区间，并复算稳定性指标。结果hash绑定后回填原树EI-2-3-3。
