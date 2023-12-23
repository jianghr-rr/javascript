/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

// This tiny example illustrates how little code is necessary build /
// train / predict from a model in TensorFlow.js.  Edit this code
// and refresh the index.html to quickly explore the API.

// Tiny TFJS train / predict example.
async function run() {
  // Create a simple model.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // 损失函数和优化器
  // 'meanAbsoluteError'是指损失函数会先计算预测（modelOutput）和目标（target）的差值，
  // 然后取其绝对值（将其变为非负数），
  // 最后返回这些绝对值（absolute）的均值（average）
  // meanAbsoluteError = average( absolute(modelOutput - targets) )
  // 'sgd'是指优化器，它通过梯度下降的方式调整模型的权重，
  // sgd是随机梯度下降算法（stochastic gradient descent）的简称
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // 以上准备好了模型

  // Generate some synthetic data for training. (y = 2x - 1)
  // 将原始的JavaScript数据结构中存储的数据转换为张量
  // 此处的[6, 1]描述的是张量的“形状”，
  // 简单来说，
  // 这里的形状意味着我们想将原数组理解为6个样本，每个样本都是1个数字。
  // 张量是将矩阵概念泛化到任意维度的结果，维度数和每个维度的尺寸叫作张量的形状（shape）
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // 以上准备好了训练集

  // 开始训练
  // TensorFlow.js可以通过调用模型的fit()方法来训练模型
  // xs和ys是训练集，epochs是训练轮数
  await model.fit(xs, ys, {epochs: 250});

  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately 39.
  document.getElementById('micro-out-div').innerText =
      model.predict(tf.tensor2d([20], [1, 1])).dataSync();
}

run();
