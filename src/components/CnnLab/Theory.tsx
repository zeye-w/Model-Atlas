const lessons = [
  ["Convolution / Cross-correlation", "y[t] = Σₖ x[t+k]w[k]", "x 为输入，w 为权重，t 为输出位置，k 为核内位置。", "深度学习的卷积层使用互相关，按权重原顺序相乘。数学卷积则反转核。", "[2, 3] 与 [4, 5] 互相关得到 2×4 + 3×5 = 23。", 0],
  ["Kernel / Stride / Padding", "Lout = ⌊(L + 2P − K)/S⌋ + 1  (D = 1)", "L 为输入长度，P 为单侧填充量，K 为核宽，S 为步幅。", "零填充扩展边界，步幅决定窗口每次移动多少。", "L=8，P=1，K=3，S=2 时，输出长度为 4。", 0],
  ["Dilation / Receptive Field", "R = 1 + (K − 1)D", "R 为单层感受野宽度，K 为核宽，D 为扩张率。", "扩张在核的采样点之间插入间隔，从而扩大感受野。", "K=3、D=2 时，采样位置为 0、2、4，覆盖宽度 5。", 0],
  ["ReLU", "a = max(0, z)", "z 为激活前的值，a 为输出。", "保留正值并把负值截为零，引入非线性。", "[-2, 0, 3] → [0, 0, 3]。", 0],
  ["Max Pooling", "y[t] = max₀≤ₖ<K x[tS+k]", "K 为池化窗口宽度，S 为步幅。", "每个窗口保留最大响应，反传时将梯度传到记录的最大值位置。", "[1, 4, 3, 2] 以 K=2、S=2 池化得到 [4, 3]。", 0],
  ["Global Average Pooling", "g[c] = (1/L) Σₜ a[c,t]", "c 为通道，t 为位置，L 为特征长度。", "沿位置维求均值，每个通道压缩成一个数。", "一个通道 [2, 4, 6] 的均值为 4。", 1],
  ["Dense Layer", "z[j] = b[j] + Σᵢ W[j,i]g[i]", "g 为特征向量，W 为权重矩阵，b 为偏置。", "每个类别的分数是输入特征的加权和加偏置。", "g=[2, 3]，W 的一行为 [1, −1]，b=0.5，得到 −0.5。", 1],
  ["Softmax", "p[j] = exp(z[j] − m) / Σᵢ exp(z[i] − m)", "z 为类别分数，m 为最大分数，p 为概率。", "指数归一化将类别分数转换为概率。减去最大值可防止指数溢出。", "[0, ln 2, 0] → [0.25, 0.50, 0.25]。", 1],
  ["Cross Entropy", "ℓ = −ln p[y]", "y 为真实类别，p[y] 为该类别的预测概率。", "模型赋予真实类别的概率越高，损失越小。", "真实类别概率为 0.25 时，损失约为 1.386。", 1],
  ["Forward Propagation", "hₗ = fₗ(hₗ₋₁; θₗ)", "h 为层输出，f 为层变换，θ 为参数，l 为层编号。", "逐层传递输入，计算类别分数与概率。", "输入 2 乘权重 −1 得 −2，经过 ReLU 得到 0。", 1],
  ["Backpropagation", "∂ℓ/∂w = (∂ℓ/∂z)(∂z/∂w)", "ℓ 为损失，z 为中间变量，w 为权重。", "用链式法则把损失的导数从输出传回各层，卷积共享权重的梯度需累加各位置贡献。", "z=3w 且 ∂ℓ/∂z=0.4 时，∂ℓ/∂w=1.2。", 1],
  ["Gradient Descent", "w′ = w − η∇wℓ", "w 为当前权重，η 为学习率，∇wℓ 为损失对权重的梯度。", "按梯度反方向更新参数。本实验用小批量样本的平均梯度执行 SGD。", "w=0.8，η=0.1，梯度为 1.2 时，更新后为 0.68。", 1],
] as const;

export function Theory({ open }: { open: (workspace: number) => void }) {
  return <div className="cnn-grid cnn-theory" lang="zh">
    {lessons.map(([title, formula, symbols, explanation, example, workspace]) => <details className="cnn-card" key={title}>
      <summary lang="en">{title}</summary>
      <p className="cnn-formula">{formula}</p>
      <p>{symbols}</p><p>{explanation}</p><p>{example}</p>
      <button onClick={() => open(workspace)}>{workspace ? "打开训练台" : "打开卷积探索器"}</button>
    </details>)}
  </div>;
}
