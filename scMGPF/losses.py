import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Bernoulli


class x_recoLoss(nn.Module):
    """
    Reconstruct the lost RNA data.
    Parameters:
        pos_weight (float): Positive sample weight, balancing sparsity (default 1.0).
        eps (float): Numerical stability constant (default 1e-10).
        l2_lambda (float): L2 regularization weight (default 1e-4).
    """

    def __init__(self, pos_weight=1.0, eps=1e-10, l2_lambda=1e-4, use_nb=False):
        super(x_recoLoss, self).__init__()
        self.pos_weight = pos_weight
        self.eps = eps
        self.l2_lambda = l2_lambda
        self.use_nb = use_nb  

    def forward(self, x, reco_x, model):
        """
        Parameters:
            x (torch.Tensor): RNA data input into the encoder, shape (batch_size, gene_dim).
            mu (torch.Tensor): Predicted mean, shape (batch_size, gene_dim).
            theta (torch.Tensor): Predict dispersion, shape (batch_size, gene_dim).
            model (nn.Module, optional): Encoder for L2 regularization.

        Returns:
            loss (torch.Tensor): Negative log-likelihood loss + L2 regularization.
        """

        reco_r_loss = F.mse_loss(reco_x, x, reduction='mean') * self.pos_weight

        l2_loss = 0.0
        if model is not None:
            l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * self.l2_lambda

        return reco_r_loss + l2_loss


class y_recoLoss(nn.Module):
    """
    Reconstruct the lost ATAC data
        Parameters:
        pos_weight (float): Positive sample weight, balancing sparsity (default 1.0).
        eps (float): Numerical stability constant (default 1e-10).
        l2_lambda (float): L2 regularization weight (default 1e-4).
    """

    def __init__(self, pos_weight=1.0, eps=1e-10, l2_lambda=1e-4, use_binary=False):
        super(y_recoLoss, self).__init__()
        self.pos_weight = pos_weight
        self.eps = eps
        self.l2_lambda = l2_lambda
        self.use_binary = use_binary

    def forward(self, y, reco_y, model):
        """
        Parameters:
            y (torch.Tensor): The ATAC data input to the encoder, shape (batch_size, peak_dim).
            p (torch.Tensor): Prediction probability, shape (batch_size, peak_dim).
            model (nn.Module, optional): An encoder for L2 regularization.

        Returns:
            loss (torch.Tensor): Weighted binary cross-entropy loss + L2 regularization.
        """
        if self.use_binary:
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError("ATAC data must be binary (0 or 1) for binary_cross_entropy")
            reco_a_loss = F.binary_cross_entropy(
                reco_y, y, reduction='mean'
            )
        else:
            reco_a_loss = F.mse_loss(reco_y, y, reduction='mean') * self.pos_weight
        l2_loss = 0.0
        if model is not None:
            l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * self.l2_lambda

        return reco_a_loss + l2_loss


class AdaptiveFeatureLinkedCosineLoss(nn.Module):
    """
    Enhanced Cosine Similarity Alignment Loss with Feature Links and Adaptive Temperature.
    - Feature Links: Weighted by biological correspondence (e.g., gene-peak matrix).
    - Adaptive Temperature: Learnable scaling based on embedding entropy.

    Parameters:
        temperature_init (float): Initial temperature (default: 0.1, from benchmarks).
        link_matrix (torch.Tensor, optional): Pre-computed feature link matrix (e.g., gene-peak overlaps), shape (latent_dim_rna, latent_dim_atac). If None, uniform weights.
        learn_temp (bool): Enable adaptive temperature (default: True).
    """

    def __init__(self, temperature_init=0.1, link_matrix=None, learn_temp=True, *args, **kwargs):
        super().__init__()
        self.temperature_init = temperature_init
        self.link_matrix = link_matrix  # Optional: (latent_dim_rna, latent_dim_atac)
        if link_matrix is not None:
            self.link_matrix = F.normalize(link_matrix.float(), dim=-1)  # Normalize for weights
        self.learn_temp = learn_temp
        if learn_temp:
            self.temp_param = nn.Parameter(torch.tensor(temperature_init))  # Learnable temp

    def forward(self, z_rna, z_atac):
        """
        Parameters:
            z_rna (torch.Tensor): RNA embeddings, shape (batch_size, latent_dim).
            z_atac (torch.Tensor): ATAC embeddings, shape (batch_size, latent_dim).

        Returns:
            loss (torch.Tensor): Enhanced alignment loss.
        """
        batch_size, latent_dim = z_rna.shape
        assert z_atac.shape == z_rna.shape, "Embeddings must have same shape"

        # Step 1: Normalize embeddings to unit vectors
        z_rna_norm = F.normalize(z_rna, p=2, dim=-1)
        z_atac_norm = F.normalize(z_atac, p=2, dim=-1)

        # Step 2: Compute base cosine similarity (paired: diagonal)
        cos_sim = torch.sum(z_rna_norm * z_atac_norm, dim=-1)  # Shape: (batch_size,)

        # Step 3: Feature Link Weighting (if provided)
        if self.link_matrix is not None:
            # Expand link_matrix to batch: (batch, latent_rna, latent_atac) -> weighted sum
            # Assume latent_dim same; for diff dims, adjust broadcasting
            link_weights = self.link_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, d, d)
            weighted_cos = torch.sum(link_weights * (z_rna_norm.unsqueeze(-1) * z_atac_norm.unsqueeze(1)),
                                     dim=(1, 2))  # (batch,)
            cos_sim = weighted_cos  # Apply weights
        else:
            # Uniform if no links
            pass

        # Step 4: Adaptive Temperature Scaling
        if self.learn_temp:
            # Adaptive: tau = sigmoid(temp_param) * init + (1 - sigmoid) * entropy-based adjust
            entropy_rna = -torch.sum(z_rna_norm * torch.log(z_rna_norm + 1e-8), dim=-1).mean()  # Avg embedding entropy
            entropy_atac = -torch.sum(z_atac_norm * torch.log(z_atac_norm + 1e-8), dim=-1).mean()
            avg_entropy = (entropy_rna + entropy_atac) / 2
            adaptive_scale = torch.sigmoid(self.temp_param) * self.temperature_init + (
                        1 - torch.sigmoid(self.temp_param)) * avg_entropy
            tau = adaptive_scale.clamp(min=0.01, max=1.0)  # Clamp for stability
        else:
            tau = self.temperature_init

        # Scaled cosine
        scaled_cos = cos_sim / tau

        # Step 5: Negative mean loss (maximize similarity)
        loss = -torch.mean(scaled_cos)

        return loss


class ContrastiveLoss(nn.Module):
    """"Contrastive learning Loss function"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, rna_features, atac_features, labels=None):
        """
        Args:
            rna_features: [batch_size, hidden_size]
            atac_features: [batch_size, hidden_size]
            labels: [batch_size] - Used for hard negative sample mining
        """
        # Normalized Features
        rna_features = F.normalize(rna_features, dim=1)
        atac_features = F.normalize(atac_features, dim=1)
        
        # Calculate the similarity matrix
        similarity_matrix = torch.matmul(rna_features, atac_features.T) / self.temperature

        # The diagonal elements are positive sample pairs
        positive_pairs = torch.diag(similarity_matrix)

        # Calculate the InfoNCE loss
        # For each RNA feature, the ATAC feature is treated as a positive sample, and all other ATAC features are treated as negative samples
        rna_to_atac_loss = -torch.log(
            torch.exp(positive_pairs) /
            torch.sum(torch.exp(similarity_matrix), dim=1)
        ).mean()

        # For each ATAC feature, RNA features are taken as positive samples, and all other RNA features are taken as negative samples
        atac_to_rna_loss = -torch.log(
            torch.exp(positive_pairs) /
            torch.sum(torch.exp(similarity_matrix.T), dim=1)
        ).mean()

        # If labels are provided, hard negative sample mining can be carried out
        if labels is not None:
            
            # Create a label mask. Samples with the same label are treated as hard negative samples
            label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            # Remove the diagonal (yourself)
            label_mask = label_mask - torch.eye(label_mask.size(0), device=label_mask.device)

            # Hard negative sample loss
            hard_negative_loss = torch.sum(
                label_mask * torch.exp(similarity_matrix), dim=1
            ) / (torch.sum(label_mask, dim=1) + 1e-8)
            hard_negative_loss = torch.log(hard_negative_loss).mean()

            return (rna_to_atac_loss + atac_to_rna_loss) / 2 + 0.1 * hard_negative_loss

        return (rna_to_atac_loss + atac_to_rna_loss) / 2


class CosineAlignmentLoss(nn.Module):
    """
    Low makes the alignment (of the same cell mode) more prominent (emphasizing alignment), and high makes the distribution smoother (reducing noise sensitivity)
    Cosine Similarity Alignment Loss, adapted from scPairing for scMGPF.
    Encourages similarity between RNA and ATAC embeddings before fusion.
    Computes negative mean cosine similarity on normalized embeddings.
    The negative value should approach -1, indicating high similarity
    Parameters:
        temperature (float, optional): Scaling factor for cosine similarity (default: 1.0, no scaling).
    """
    def __init__(self, temperature=0.1):
        super(CosineAlignmentLoss, self).__init__()
        self.temperature = temperature
        # Learnable log-temperature for adaptive scaling (ensures temperature > 0)
        # self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, z_rna, z_atac):
        """
        Parameters:
            z_rna (torch.Tensor): RNA embeddings from GraphSAGEEncoder, shape (batch_size, latent_dim).
            z_atac (torch.Tensor): ATAC embeddings from GATv2Encoder, shape (batch_size, latent_dim).
        
        Returns:
            loss (torch.Tensor): Negative mean cosine similarity loss.
        """

        # Normalize embeddings to unit vectors (L2 norm)
        z_rna_norm = F.normalize(z_rna, p=2, dim=-1)
        z_atac_norm = F.normalize(z_atac, p=2, dim=-1)

        # Compute cosine similarity (dot product after normalization)
        # temperature = torch.exp(self.log_temperature)
        # For paired cells (diagonal of similarity matrix)
        cos_sim = torch.sum(z_rna_norm * z_atac_norm, dim=-1) / self.temperature
        
        # Negative mean: minimize loss to maximize similarity
        loss = -torch.mean(cos_sim)
        
        return loss


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = nn.Parameter(torch.Tensor((mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] /
                         (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def Total_Loss(reco_r_loss, reco_a_loss,
               cos_loss,
               # gw_loss,
               contra_loss,
               nb_model, ber_model, epoch=None,
               nb_weight=0.5, ber_weight=0.7,
               cos_weight=0.7,
               gw_weight=0.05, contra_weight=0.8,
               weight_decay=0.95, max_norm=1.0):
    """
    Calculate the joint loss, combined reconfiguration, graph structure, GW and contrastive loss.
    parameters:
        nb_loss (NBReconstructionLoss): RNA reconstruction loss.
        bernoulli_loss (BernoulliReconstructionLoss): ATAC reconstruction loss.
        graph_loss (GraphStructureLoss): GraphStructureLoss.
        gw_loss (GromovWassersteinLoss): GW loss.
        contra_loss (ContrastiveLoss):Contrastive loss.
        nb_model (nn.Module, optional): NBDecoder model.
        bernoulli_model (nn.Module, optional): BernoulliDecoder model.
        epoch (int): The current number of training rounds is used for dynamic weights.
        gw_weight (float): Initial GW loss weight (default 0.1).
        graph_weight (float): Initial graph structure loss weight (default 0.1).
        contra_weight (float): Initial comparison loss weight (default 0.1).
        weight_decay (float): Weight decay factor (default 0.9, decaying every 10 epochs).
        max_norm (float): Gradient clipping maximum norm (default 1.0).


    Returns:
        total_loss (torch.Tensor): Joint loss.
        loss_dict (dict): The values of each loss component.
    """
    # Dynamically adjust weights               
    decay_factor = weight_decay ** (epoch // 10)
    nb_weight = nb_weight
    ber_weight = ber_weight
    # gw_weight = gw_weight * decay_factor
    cos_weight = cos_weight * decay_factor
    contra_weight = contra_weight * decay_factor

    # Joint Loss
    total_loss = (
            nb_weight * reco_r_loss +
            ber_weight * reco_a_loss +
            cos_weight * cos_loss +
            # gw_weight * gw_loss +
            contra_weight * contra_loss
    )

    # Gradient Clipping
    if nb_model is not None and ber_model is not None:
        torch.nn.utils.clip_grad_norm_(
            list(nb_model.parameters()) + list(ber_model.parameters()),
            max_norm=max_norm
        )
    # Record the amount of loss
    loss_dict = {
        'epoch': epoch,
        'reco_r_loss': reco_r_loss.item(),
        'reco_a_loss': reco_a_loss.item(),
        # 'gw_loss': gw_loss.item(),
        'cos_loss': cos_loss.item(),
        'contra_loss': contra_loss.item(),
        'nb_weight': nb_weight,
        'ber_weight': ber_weight,
        'cos_weight': cos_weight,
        # 'gw_weight': gw_weight,
        'contra_weight': contra_weight
    }

    return total_loss, loss_dict


