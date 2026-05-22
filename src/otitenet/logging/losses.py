from torch import nn
import torch
import torch.nn.functional as F


class ArcFaceLossWithHSM(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.5, topk=0.2):
        super(ArcFaceLossWithHSM, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        self.top_k = top_k  # Fraction of hard samples to mine
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(embeddings, weight)  # [batch_size, num_classes]
        
        # Apply ArcFace margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        logits = cosine * 1.0
        logits.scatter_(1, labels.view(-1, 1).long(), target_logit)
        logits *= self.s
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels.to(self.device), reduction='none')  # [batch_size]

        # Hard Sample Mining: Select top-k hardest samples
        if self.topk > 0:
            k = int(self.topk * len(loss))
            hard_loss, _ = torch.topk(loss, k=k)
            loss = hard_loss.mean()  # Average loss over hard samples
        else:
            loss = loss.mean()  # Average loss over all samples
        
        return loss


class ArcFaceLossWithSubcenters(nn.Module):
    def __init__(self, embedding_size, num_classes, num_subcenters=3, s=30.0, m=0.5, device="cuda"):
        super(ArcFaceLossWithSubcenters, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters  # Number of subcenters per class
        self.device = device  # Device (e.g., "cuda" or "cpu")
        
        # Initialize weights for subcenters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * num_subcenters, embedding_size).to(self.device))
        nn.init.xavier_uniform_(self.weight)  # Initialize weights

    def get_logits(self, embeddings):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity between embeddings and all subcenters
        cosine = F.linear(embeddings, weight)  # [batch_size, num_classes * num_subcenters]
        
        # Reshape cosine to group subcenters by class
        cosine = cosine.view(-1, self.num_classes, self.num_subcenters)  # [batch_size, num_classes, num_subcenters]
        
        # Find the maximum cosine similarity for each class (best subcenter)
        cosine, subcenters = torch.max(cosine, dim=2)  # [batch_size, num_classes]
        # cosine_argmax = torch.argmax(cosine, dim=1)  # [batch_size, 1]
        # Get the correct subcenter according to cosine_argmax
        subcenters = subcenters[:, 1]
        # Apply the margin to the correct class logits
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))  # Clamp to avoid NaN
        target_logit = torch.cos(theta + self.m)  # Add angular margin
        logits = cosine * 1.0  # Copy cosine values
        return logits, cosine, target_logit, subcenters

    def forward(self, embeddings, labels):
        # Get the logits for each class
        logits, cosine, target_logit, subcenters = self.get_logits(embeddings)
        # Convert labels to one-hot encoding and move to the same device
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)  # [batch_size, num_classes]
        logits = logits - one_hot_labels * (cosine - target_logit)  # Apply margin only to the correct class
        
        # Scale the logits

        logits *= self.s
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels.to(self.device))
        return loss, subcenters


class ArcFaceLossWithSubcentersHSM(nn.Module):
    def __init__(self, embedding_size, num_classes, num_subcenters=3, s=30.0, m=0.5, topk=0.2, device="cuda"):
        super(ArcFaceLossWithSubcentersHSM, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters  # Number of subcenters per class
        self.device = device  # Device (e.g., "cuda" or "cpu")
        self.topk = topk
        
        # Initialize weights for subcenters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * num_subcenters, embedding_size).to(self.device))
        nn.init.xavier_uniform_(self.weight)  # Initialize weights

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity between embeddings and all subcenters
        cosine = F.linear(embeddings, weight)  # [batch_size, num_classes * num_subcenters]
        
        # Reshape cosine to group subcenters by class
        cosine = cosine.view(-1, self.num_classes, self.num_subcenters)  # [batch_size, num_classes, num_subcenters]
        
        # Find the maximum cosine similarity for each class (best subcenter)
        cosine, _ = torch.max(cosine, dim=2)  # [batch_size, num_classes]
        
        # Convert labels to one-hot encoding and move to the same device
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)  # [batch_size, num_classes]
        
        # Apply the margin to the correct class logits
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))  # Clamp to avoid NaN
        target_logit = torch.cos(theta + self.m)  # Add angular margin
        logits = cosine * 1.0  # Copy cosine values
        logits = logits - one_hot_labels * (cosine - target_logit)  # Apply margin only to the correct class
        
        # Scale the logits
        logits *= self.s
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels.to(self.device), reduction='none')  # [batch_size]

        # Hard Sample Mining: Select top-k hardest samples
        if self.topk > 0:
            k = int(self.topk * len(loss))
            hard_loss, _ = torch.topk(loss, k=k)
            loss = hard_loss.mean()  # Average loss over hard samples
        else:
            loss = loss.mean()  # Average loss over all samples
        return loss
        

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.5, device="cuda"):
        super(ArcFaceLoss, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        self.num_classes = num_classes
        self.device = device  # Device (e.g., "cuda" or "cpu")
        
        # Initialize weights and move to the specified device
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size)).to(self.device)
        nn.init.xavier_uniform_(self.weight)  # Initialize weights

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(embeddings, weight)  # [batch_size, num_classes]
        
        # Convert labels to one-hot encoding and move to the same device
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(self.device)  # [batch_size, num_classes]
        
        # Apply the margin to the correct class logits
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))  # Clamp to avoid NaN
        target_logit = torch.cos(theta + self.m)  # Add angular margin
        logits = cosine * 1.0  # Copy cosine values
        logits = logits - one_hot_labels * (cosine - target_logit)  # Apply margin only to the correct class
        
        # Scale the logits
        logits *= self.s
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels.to(self.device))
        return loss


class SoftmaxContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, distance_metric='cosine'):
        """
        Softmax Contrastive Loss

        Args:
        - temperature: scaling factor for the softmax (higher = softer, lower = sharper).
        - distance_metric: 'cosine' or 'euclidean' (cosine works better in many cases).
        """
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric

    def forward(self, anchor, positive, negatives):
        """
        Compute loss given anchor, positive, and multiple negatives.

        Args:
        - anchor: Tensor of shape (batch_size, embedding_dim)
        - positive: Tensor of shape (batch_size, embedding_dim)
        - negatives: Tensor of shape (batch_size, num_negatives, embedding_dim)

        Returns:
        - loss: Computed Softmax Contrastive Loss
        """
        if self.distance_metric == 'cosine':
            # Cosine similarity (higher is better)
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
            negatives = F.normalize(negatives, p=2, dim=-1)
            pos_sim = (anchor * positive).sum(dim=-1)  # (batch_size,)
            neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=-1)  # (batch_size, num_negatives)
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (lower is better, so we negate it for softmax)
            pos_sim = -torch.norm(anchor - positive, dim=-1)  # (batch_size,)
            neg_sim = -torch.norm(anchor.unsqueeze(1) - negatives, dim=-1)  # (batch_size, num_negatives)
        elif self.distance_metric == 'manhattan':
            # Manhattan distance (lower is better, so we negate it for softmax)
            pos_sim = -torch.norm(anchor - positive, p=1, dim=-1)
            neg_sim = -torch.norm(anchor.unsqueeze(1) - negatives, p=1, dim=-1)
        elif self.distance_metric == 'chebyshev':
            # Chebyshev distance (lower is better, so we negate it for softmax)
            pos_sim = -torch.norm(anchor - positive, p=float('inf'), dim=-1)
            neg_sim = -torch.norm(anchor.unsqueeze(1) - negatives, p=float('inf'), dim=-1)
        elif self.distance_metric == 'minkowski':
            # Minkowski distance (lower is better, so we negate it for softmax)
            pos_sim = -torch.norm(anchor - positive, p=3, dim=-1)
            neg_sim = -torch.norm(anchor.unsqueeze(1) - negatives, p=3, dim=-1)
        elif self.distance_metric == 'SNR':
            # Signal-to-Noise Ratio (higher is better)
            pos_sim = torch.var(anchor - positive) / torch.var(anchor)
            neg_sim = torch.var(anchor.unsqueeze(1) - negatives, dim=-1) / torch.var(anchor.unsqueeze(1), dim=-1)
        else:
            raise ValueError("Unsupported distance metric. Choose 'cosine' or 'euclidean'.")

        # Combine similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_negatives)

        # Scale with temperature
        logits /= self.temperature

        # Compute softmax loss
        loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
        return loss


class TupletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TupletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positives, negatives):
        """
        Args:
            anchor: Embedding of the anchor sample.
            positives: List of embeddings of positive samples.
            negatives: List of embeddings of negative samples.
        """
        # Compute distances between anchor and positives
        pos_distances = [torch.norm(anchor - pos, p=2) for pos in positives]
        pos_distance = torch.mean(torch.stack(pos_distances))

        # Compute distances between anchor and negatives
        neg_distances = [torch.norm(anchor - neg, p=2) for neg in negatives]
        neg_distance = torch.mean(torch.stack(neg_distances))

        # Compute Tuplet Loss
        loss = torch.relu(pos_distance - neg_distance + self.margin)
        return loss

