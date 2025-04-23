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
        x_input = torch.unsqueeze(x, dim=0)
        original_z = self.model.model.encoder(x_input).squeeze()
    
    # 2. 创建可优化的代理特征向量
    proxy_z = original_z.detach().clone()
    proxy_z.requires_grad_(True)
    
    # 3. 设置优化器
    if self.cert_optimizer == "sgd":
        optimizer = torch.optim.SGD([proxy_z], lr=lr)
    elif self.cert_optimizer == "adam":
        optimizer = torch.optim.Adam([proxy_z], lr=lr)
    
    # 4. 存储测地线路径上的点
    geodesic_path = [original_z.clone()]
    
    # 5. 优化过程 - 找到最优的代理特征向量
    for i in range(step):
        # 前向传播
        pred = self.model.model.g(proxy_z.unsqueeze(0))
        loss = self.criterion(pred, onehot_label.unsqueeze(0))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 存储当前点以构建测地线路径
        if (i+1) % (step // 10) == 0 or i == 0:  # 存储10个关键点以节省内存
            geodesic_path.append(proxy_z.detach().clone())
        
        # 打印优化进度
        if (i+1) % (step // 5) == 0:
            print(f"优化步骤 {i+1}/{step}, 损失: {loss.item():.6f}")
    
    # 6. 计算测地距离 - 沿路径积分
    total_distance = 0.0
    
    # 在关键点之间插值以获得更平滑的测地线
    refined_path = []
    for i in range(len(geodesic_path) - 1):
        start_point = geodesic_path[i]
        end_point = geodesic_path[i+1]
        
        # 在两点之间线性插值
        num_interp = 10  # 每段插入10个点
        for t in range(num_interp + 1):
            interp_factor = t / num_interp
            interp_point = start_point * (1 - interp_factor) + end_point * interp_factor
            refined_path.append(interp_point)
    
    # 7. 沿路径计算局部黎曼度量并积分距离
    for i in range(1, len(refined_path)):
        prev_z = refined_path[i-1]
        curr_z = refined_path[i]
        
        # 计算欧式位移向量
        displacement = curr_z - prev_z
        
        # 估计该点的局部黎曼度量
        local_metric = self.estimate_local_riemann_metric(
            (prev_z * (1-alpha) + curr_z * alpha).unsqueeze(0)  # 在两点之间的加权位置估计度量
        )
        
        # 在黎曼度量下计算线元长度
        # ds^2 = dx^T G dx
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
    
    return total_distance, original_z, proxy_z

def estimate_local_riemann_metric(self, z, epsilon=1e-4):
    """
    估计特征空间中的局部黎曼度量张量
    使用输出空间中的Fisher信息矩阵近似
    
    参数:
    z: 特征空间中的点
    epsilon: 用于数值计算梯度的扰动大小
    
    返回:
    局部黎曼度量张量 (Fisher信息矩阵)
    """
    z_dim = z.shape[1]
    batch_size = z.shape[0]
    
    try:
        # 计算当前点的模型输出
        with torch.no_grad():
            base_output = self.model.model.g(z)
        
        # 初始化雅可比矩阵
        # 输出空间维度 × 特征空间维度
        output_dim = base_output.shape[1]
        jacobian = torch.zeros((batch_size, output_dim, z_dim), device=z.device)
        
        # 对每个特征维度计算数值梯度
        for i in range(z_dim):
            # 创建扰动向量
            h = torch.zeros_like(z)
            h[:, i] = epsilon
            
            # 前向差分计算梯度
            with torch.no_grad():
                z_plus_h = z + h
                output_plus_h = self.model.model.g(z_plus_h)
                
                # 数值梯度
                jacobian[:, :, i] = (output_plus_h - base_output) / epsilon
        
        # 计算Fisher信息矩阵: J^T * J
        batch_metrics = []
        for b in range(batch_size):
            fisher_info = torch.matmul(jacobian[b].t(), jacobian[b])
            
            # 确保度量张量是正定的
            eigenvalues = torch.linalg.eigvalsh(fisher_info)
            min_eig = torch.min(eigenvalues)
            
            # 如果有非正特征值，添加小的扰动
            if min_eig <= 0:
                fisher_info += torch.eye(z_dim, device=z.device) * (abs(min_eig) + 1e-6)
                
            batch_metrics.append(fisher_info)
            
        # 返回批次中第一个(或唯一)度量张量
        return batch_metrics[0]
        
    except Exception as e:
        print(f"估计黎曼度量时出错: {e}")
        return None

def visualize_geodesic_path(self, original_z, proxy_z, num_points=20):
    """
    在特征空间中可视化测地线路径
    使用PCA或t-SNE降维到2D或3D进行可视化
    
    参数:
    original_z: 原始特征向量
    proxy_z: 代理特征向量
    num_points: 路径上的插值点数
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # 生成测地线路径上的点
        path_points = []
        for t in torch.linspace(0, 1, num_points):
            # 在特征空间中线性插值
            interp_z = original_z * (1 - t) + proxy_z * t
            path_points.append(interp_z.cpu().detach().numpy())
        
        # 将路径点堆叠为一个数组
        path_array = np.stack(path_points)
        
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        path_2d = pca.fit_transform(path_array)
        
        # 可视化2D路径
        plt.figure(figsize=(10, 8))
        plt.scatter(path_2d[:, 0], path_2d[:, 1], c=np.arange(num_points), cmap='viridis')
        plt.plot(path_2d[:, 0], path_2d[:, 1], 'r--')
        plt.scatter(path_2d[0, 0], path_2d[0, 1], c='blue', s=100, label='Original Feature')
        plt.scatter(path_2d[-1, 0], path_2d[-1, 1], c='red', s=100, label='Proxy Feature')
        plt.colorbar(label='Path Position')
        plt.legend()
        plt.title('Geodesic Path in Feature Space (PCA Projection)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.savefig('geodesic_path.png')
        plt.close()
        
        print(f"测地线路径可视化已保存为 'geodesic_path.png'")
        
    except Exception as e:
        print(f"可视化测地线路径时出错: {e}")

# 改进可视化的建议
# 如果您想进一步增强可视化效果，可以考虑：

# 使用t-SNE或UMAP等非线性降维方法，可能比PCA更好地保留高维空间的局部结构
# 添加动态可视化，如生成GIF动画展示特征向量的变化过程
# 在路径上添加等距离标记，显示测地距离的累积
# 绘制3D可视化(使用PCA的前三个主成分)，提供更丰富的空间信息