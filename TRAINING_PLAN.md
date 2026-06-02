# AI Infra 知识蒸馏项目规划（工业级版本）

> **项目目标**
>
> 构建面向 AI Infra / 音视频 / 流媒体 / 分布式系统 / GPU 加速领域的专业问答模型。
>
> 从「能跑通 LoRA」升级到「具备实际生产价值的领域模型」。
>
> 当前基线：
>
> * Base Model：Qwen2.5-0.5B
> * Training：LoRA
> * Dataset：432 条
> * 训练方式：SFT

---

# 一、项目总体路线

```text
数据建设
    ↓
SFT训练
    ↓
自动评测
    ↓
人工评测
    ↓
错误回注
    ↓
持续迭代
```

核心原则：

```text
数据质量 > 数据数量 > 评测体系 > 训练技巧 > 超参数
```

---

# 二、阶段一：数据工程（最高优先级）

## 2.1 数据目标

当前：

```text
432 Samples
```

目标：

```text
Phase1:
432 → 2000

Phase2:
2000 → 5000

Phase3:
5000 → 10000+
```

---

## 2.2 数据分类设计

建议建立标准化分类体系：

```yaml
AI_INFRA:
  - 模型训练
  - 推理优化
  - vLLM
  - TensorRT
  - llama.cpp
  - Triton

AUDIO_VIDEO:
  - WebRTC
  - RTMP
  - RTP
  - FFmpeg
  - H264
  - H265
  - AV1

CPP:
  - C++11
  - C++17
  - STL
  - 内存管理
  - 多线程

SYSTEM:
  - Linux
  - 网络编程
  - epoll
  - io_uring
  - TCP/IP

GPU:
  - CUDA
  - Tensor Core
  - 显存优化
  - Kernel优化

ARCHITECTURE:
  - 微服务
  - 高并发
  - 分布式
  - 消息队列

INTERVIEW:
  - 基础题
  - 场景题
  - 项目题
  - 系统设计题
```

---

## 2.3 数据质量等级

新增字段：

```json
{
  "instruction": "...",
  "thinking": "...",
  "output": "...",
  "quality": "gold"
}
```

等级定义：

### Gold

领域专家审核

```text
准确率：
100%

优先训练
```

---

### Silver

GPT4 / Claude生成

人工校验

```text
准确率：
95%以上
```

---

### Bronze

自动生成

未审核

```text
仅用于扩充覆盖度
```

---

## 2.4 数据审核标准

每周抽查：

```text
随机抽取：

20~50条
```

检查：

### Thinking

```text
逻辑链完整

推导合理

无明显事实错误
```

---

### Output

```text
回答准确

无幻觉

结论正确
```

---

## 2.5 数据扩展来源

### P0

Claude Opus

GPT-4.1

Gemini 2.5 Pro

生成高质量问答

目标：

```text
+1000条
```

---

### P1

官方文档抽取

例如：

* FFmpeg
* WebRTC
* Kubernetes
* PyTorch
* CUDA

构造：

```text
文档
↓
QA
↓
人工审核
```

---

### P2

真实面试题

来源：

```text
牛客网

Boss

脉脉

Github面试仓库
```

---

### P3（最重要）

错误回注

```text
用户提问
↓
模型答错
↓
修正答案
↓
加入训练集
```

---

# 三、阶段二：数据集划分

禁止全部数据参与训练。

必须拆分：

```text
Train
Validation
Test
```

推荐：

```yaml
train: 80%
valid: 10%
test: 10%
```

例如：

```text
2000条

1600 train

200 valid

200 test
```

---

## 3.1 冻结测试集

原则：

```text
测试集永久冻结
```

禁止：

```text
加入训练

参与数据增强

参与DPO
```

---

# 四、阶段三：SFT训练

---

## 4.1 基座模型选择

### 第一阶段

```text
Qwen2.5-0.5B
```

适用于：

```text
<2000条数据
```

---

### 第二阶段

```text
Qwen2.5-1.5B
```

适用于：

```text
2000+
```

---

### 第三阶段

```text
Qwen3-1.7B Thinking
```

适用于：

```text
思维链训练
```

---

## 4.2 推荐LoRA配置

### 小数据集（<2000）

```yaml
lora:
  r: 8
  alpha: 16
  dropout: 0.1

target_modules:
  - q_proj
  - v_proj
```

---

### 中型数据集（2000~5000）

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
```

---

## 4.3 推荐训练配置

```yaml
learning_rate: 2e-4

weight_decay: 0.01

max_grad_norm: 1.0

warmup_ratio: 0.05

lr_scheduler_type: cosine
```

---

## 4.4 Epoch策略

不固定Epoch。

采用：

```yaml
EarlyStopping
```

配置：

```yaml
early_stopping_patience: 3
```

推荐：

```yaml
max_epochs: 50
```

实际通常：

```text
15~30 epoch
```

结束。

---

# 五、阶段四：防止过拟合

---

## 5.1 核心措施

### Dropout

```yaml
lora_dropout: 0.1
```

---

### Weight Decay

```yaml
weight_decay: 0.01
```

---

### Gradient Clipping

```yaml
max_grad_norm: 1.0
```

---

### NEFTune

```yaml
neftune_noise_alpha: 5
```

---

### Validation监控

观察：

```text
Train Loss

Validation Loss
```

出现：

```text
Train下降

Valid上升
```

立即停止训练。

---

# 六、阶段五：评测体系

---

## 6.1 建立标准评测集

推荐：

```text
200题
```

覆盖：

```yaml
WebRTC: 30

FFmpeg: 30

CUDA: 30

AI Infra: 40

Linux: 30

系统设计: 40
```

---

## 6.2 自动评测

### PPL

目标：

```text
Train/Test Gap < 5
```

---

### BERTScore

目标：

```text
>0.8
```

仅作辅助指标。

---

## 6.3 LLM-as-Judge

推荐：

* Claude
* GPT
* Gemini

评分维度：

```yaml
准确性:
  weight: 40%

完整性:
  weight: 25%

专业性:
  weight: 20%

结构化:
  weight: 10%

简洁性:
  weight: 5%
```

---

## 6.4 人工评测

随机：

```text
50题
```

对比：

```text
Base Model

VS

LoRA Model
```

统计：

```text
Win

Lose

Tie
```

---

# 七、阶段六：DPO优化

SFT不是终点。

推荐流程：

```text
SFT
↓
评测
↓
构造偏好数据
↓
DPO
↓
发布
```

---

## 7.1 DPO数据格式

```json
{
  "prompt": "什么是WebRTC？",
  "chosen": "正确答案",
  "rejected": "错误答案"
}
```

来源：

```text
模型历史错误回答
```

---

## 7.2 DPO目标

提升：

```text
回答质量

结构化能力

稳定性

专业度
```

---

# 八、阶段七：错误回注飞轮

这是项目长期价值核心。

---

## 流程

```text
用户提问
      ↓
模型回答
      ↓
人工审核
      ↓
发现错误
      ↓
修正答案
      ↓
加入训练集
      ↓
重新训练
```

---

## 数据来源优先级

```text
真实用户问题
    >
真实面试题
    >
Claude生成
    >
自动生成
```

---

# 九、里程碑规划

## Week 1

数据治理

```text
完成分类体系

完成质量标注

扩充到2000条
```

---

## Week 2

训练优化

```text
建立Train/Valid/Test

接入EarlyStopping

完成Baseline训练
```

---

## Week 3

评测体系

```text
建立200题评测集

接入LLM Judge

完成评测报告
```

---

## Week 4

DPO优化

```text
构建偏好数据

完成第一版DPO

发布V1模型
```

---

# 十、最终目标（V1）

```text
数据规模：
5000+

领域覆盖：
AI Infra
音视频
WebRTC
FFmpeg
Linux
CUDA
系统设计

训练方式：
SFT + DPO

评测：
LLM Judge
人工评测

部署：
Ollama
vLLM
llama.cpp

最终产出：
AI Infra 面试助手模型
音视频工程师助手模型
企业级知识蒸馏模型
```

---

# 当前最值得投入的事项（ROI最高）

按优先级排序：

```text
1. 数据量扩充到2000条
2. 建立200题标准评测集
3. 接入LLM-as-Judge
4. 引入EarlyStopping
5. 错误回注机制
6. DPO训练
7. 升级1.5B模型
8. 超参数微调
```

对于你目前的 432 条数据项目，**最优策略不是继续研究 LoRA 参数，而是先把高质量数据做到 2000~5000 条，再建立标准评测体系。** 这样后续每一次训练迭代都能被量化验证，模型能力提升也会更加稳定。
