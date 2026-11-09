/** Lesson cards for the CNN lab's Theory tab. Formulas are LaTeX, rendered on the server. */
export interface CnnLesson { title: string; latex: string; symbols: string; explanation: string; example: string; workspace: 0 | 1 }

export const cnnLessons: CnnLesson[] = [
  { title: "Convolution / Cross-correlation", latex: String.raw`y[t] = \sum_{k} x[t+k]\, w[k]`, symbols: "x 为输入，w 为权重，t 为输出位置，k 为核内位置。", explanation: "深度学习的卷积层使用互相关，按权重原顺序相乘。数学卷积则反转核。", example: "[2, 3] 与 [4, 5] 互相关得到 2×4 + 3×5 = 23。", workspace: 0 },
  { title: "Kernel / Stride / Padding", latex: String.raw`L_{\text{out}} = \left\lfloor \frac{L + 2P - K}{S} \right\rfloor + 1 \quad (D = 1)`, symbols: "L 为输入长度，P 为单侧填充量，K 为核宽，S 为步幅。", explanation: "零填充扩展边界，步幅决定窗口每次移动多少。", example: "L=8，P=1，K=3，S=2 时，输出长度为 4。", workspace: 0 },
  { title: "Dilation / Receptive Field", latex: String.raw`R = 1 + (K - 1)D`, symbols: "R 为单层感受野宽度，K 为核宽，D 为扩张率。", explanation: "扩张在核的采样点之间插入间隔，从而扩大感受野。", example: "K=3、D=2 时，采样位置为 0、2、4，覆盖宽度 5。", workspace: 0 },
  { title: "ReLU", latex: String.raw`a = \max(0, z)`, symbols: "z 为激活前的值，a 为输出。", explanation: "保留正值并把负值截为零，引入非线性。", example: "[-2, 0, 3] → [0, 0, 3]。", workspace: 0 },
  { title: "Max Pooling", latex: String.raw`y[t] = \max_{0 \le k < K} x[tS + k]`, symbols: "K 为池化窗口宽度，S 为步幅。", explanation: "每个窗口保留最大响应，反传时将梯度传到记录的最大值位置。", example: "[1, 4, 3, 2] 以 K=2、S=2 池化得到 [4, 3]。", workspace: 0 },
  { title: "Global Average Pooling", latex: String.raw`g[c] = \frac{1}{L} \sum_{t} a[c, t]`, symbols: "c 为通道，t 为位置，L 为特征长度。", explanation: "沿位置维求均值，每个通道压缩成一个数。", example: "一个通道 [2, 4, 6] 的均值为 4。", workspace: 1 },
  { title: "Dense Layer", latex: String.raw`z[j] = b[j] + \sum_{i} W[j, i]\, g[i]`, symbols: "g 为特征向量，W 为权重矩阵，b 为偏置。", explanation: "每个类别的分数是输入特征的加权和加偏置。", example: "g=[2, 3]，W 的一行为 [1, −1]，b=0.5，得到 −0.5。", workspace: 1 },
  { title: "Softmax", latex: String.raw`p[j] = \frac{\exp(z[j] - m)}{\sum_{i} \exp(z[i] - m)}`, symbols: "z 为类别分数，m 为最大分数，p 为概率。", explanation: "指数归一化将类别分数转换为概率。减去最大值可防止指数溢出。", example: "[0, ln 2, 0] → [0.25, 0.50, 0.25]。", workspace: 1 },
  { title: "Cross Entropy", latex: String.raw`\ell = -\ln p[y]`, symbols: "y 为真实类别，p[y] 为该类别的预测概率。", explanation: "模型赋予真实类别的概率越高，损失越小。", example: "真实类别概率为 0.25 时，损失约为 1.386。", workspace: 1 },
  { title: "Forward Propagation", latex: String.raw`h_l = f_l(h_{l-1};\, \theta_l)`, symbols: "h 为层输出，f 为层变换，θ 为参数，l 为层编号。", explanation: "逐层传递输入，计算类别分数与概率。", example: "输入 2 乘权重 −1 得 −2，经过 ReLU 得到 0。", workspace: 1 },
  { title: "Backpropagation", latex: String.raw`\frac{\partial \ell}{\partial w} = \frac{\partial \ell}{\partial z}\,\frac{\partial z}{\partial w}`, symbols: "ℓ 为损失，z 为中间变量，w 为权重。", explanation: "用链式法则把损失的导数从输出传回各层，卷积共享权重的梯度需累加各位置贡献。", example: "z=3w 且 ∂ℓ/∂z=0.4 时，∂ℓ/∂w=1.2。", workspace: 1 },
  { title: "Gradient Descent", latex: String.raw`w' = w - \eta\, \nabla_{w} \ell`, symbols: "w 为当前权重，η 为学习率，∇wℓ 为损失对权重的梯度。", explanation: "按梯度反方向更新参数。本实验用小批量样本的平均梯度执行 SGD。", example: "w=0.8，η=0.1，梯度为 1.2 时，更新后为 0.68。", workspace: 1 },
];
