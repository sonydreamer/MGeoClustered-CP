from torch.autograd import function as F
F.jacobian()


def compute_feature_space_geodesic(self, x, onehot_label, step=100, lr=0.01, alpha=0.5):
    """
    计算原始特征向量与优化后的"代理特征向量"间的测地距离

    参数:
    x: 输入图像
    onehot_label: 目标类别的独热编码
    step: 优化迭代次数
    lr: 学习率
    alpha: 测地线离散化插值参数，控制度量估计点在路径上的位置，alpha=0表示前一点，alpha=1表示当前点，alpha=0.5表示中点

    返回:
    原始特征向量和代理特征向量之间的测地距离
    """
    self.model.model.eval()
    x = x.to(self.model.device)
    onehot_label = onehot_label.to(self.model.device)

    # 1. 获取原始特征向量
    with torch.no_grad():
        x_input = x.unsqueeze(0)
        original_z = self.model.model.encoder(x_input).squeeze()

    # 2. 创建可优化的代理特征向量
    proxy_z = original_z.detach().clone().requires_grad_(True)

    # 3. 设置优化器
    if self.cert_optimizer == "sgd":
        optimizer = torch.optim.SGD([proxy_z], lr=lr)
    elif self.cert_optimizer == "adam":
        optimizer = torch.optim.Adam([proxy_z], lr=lr)

    # 4. 优化过程 - 使用增量计算测地距离而不是存储路径
    prev_z = original_z.clone()
    total_distance = 0.0

    # 预编译模型前向传播函数以提高速度
    try:
        model_g = torch.jit.script(self.model.model.g)
    except:
        model_g = self.model.model.g

    for i in range(step):
        # 前向传播
        pred = model_g(proxy_z.unsqueeze(0))
        loss = self.criterion(pred, onehot_label.unsqueeze(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算当前点与前一点之间的距离增量
        curr_z = proxy_z.detach()  # 不需要.clone()，减少内存使用
        displacement = curr_z - prev_z

        # 仅在特定迭代步骤计算度量和距离（可减少计算量）
        # 对于大的step值，可以进一步减小计算频率
        if (i+1) % max(1, step // 20) == 0 or i == step - 1:
            # 计算插值点用于度量估计
            interp_z = (prev_z * (1-alpha) + curr_z * alpha).unsqueeze(0)
            
            # 估计该点的局部黎曼度量
            local_metric = self.estimate_local_riemann_metric_vectorized(interp_z)

            # 在黎曼度量下计算线元长度
            if local_metric is not None:
                # 转换到适当的形状以进行矩阵运算
                disp_flat = displacement.view(-1, 1)  # 列向量
                local_distance = torch.sqrt(
                    torch.matmul(torch.matmul(disp_flat.t(), local_metric), disp_flat)
                ).item()
            else:
                # 如果度量估计失败，回退到欧几里得距离
                local_distance = torch.norm(displacement, p=2).item()

            total_distance += local_distance
        
        # 更新前一点
        prev_z = curr_z

    return total_distance

def estimate_local_riemann_metric_vectorized(self, z, epsilon=1e-4):
    """
    使用PyTorch的autograd机制估计特征空间中的局部黎曼度量张量
    
    参数:
    z: 特征空间中的点
    epsilon: 用于数值计算的容差

    返回:
    局部黎曼度量张量 (Fisher信息矩阵)
    """
    z_dim = z.shape[1]
    batch_size = z.shape[0]
    
    try:
        # 创建一个需要梯度的z副本
        z_grad = z.clone().detach().requires_grad_(True)
        
        # 前向传播
        output = self.model.model.g(z_grad)
        output_dim = output.shape[1]
        
        # 初始化雅可比矩阵
        jacobian = torch.zeros((batch_size, output_dim, z_dim), device=z.device)
        
        # 使用autograd计算雅可比矩阵 - 比逐维度数值计算更高效
        for i in range(output_dim):
            # 清零梯度
            if z_grad.grad is not None:
                z_grad.grad.zero_()
            
            # 创建one-hot向量以选择输出维度
            grad_output = torch.zeros_like(output)
            grad_output[:, i] = 1.0
            
            # 反向传播
            output.backward(grad_output, retain_graph=(i < output_dim - 1))
            
            # 存储梯度
            if z_grad.grad is not None:
                jacobian[:, i, :] = z_grad.grad
        
        # 计算Fisher信息矩阵: J^T * J
        batch_metrics = []
        for b in range(batch_size):
            # 使用PyTorch的批处理矩阵乘法
            fisher_info = torch.matmul(jacobian[b].t(), jacobian[b])
            
            # 确保度量张量是正定的
            eigenvalues = torch.linalg.eigvalsh(fisher_info)
            min_eig = torch.min(eigenvalues)
            
            if min_eig <= 0:
                fisher_info += torch.eye(z_dim, device=z.device) * (abs(min_eig) + 1e-6)
            
            batch_metrics.append(fisher_info)
        
        return batch_metrics[0]
    
    except Exception as e:
        print(f"估计黎曼度量时出错: {e}")
        return None