import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from model.clip.clip import load
import math
import logging
import pdb
from sklearn import metrics
from trainer.metrics.base_metrics_class import calculate_metrics_for_train
from peft import get_peft_model, LNTuningConfig
from .prompt_learner import tokenize, PromptLearner, TextEncoder


_logger = logging.getLogger(__name__)

class ClipFeatureHead(nn.Module):
    def __init__(self, num_quires, embed_dim, normalize_features=True):
        super().__init__()
        self.normalize_features = normalize_features
        self.first_process = nn.Sequential(
            nn.Conv1d(in_channels=num_quires, out_channels=num_quires // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_quires // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_quires // 2, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.third_process = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=768),
            nn.ReLU()
        )
        self.second_process = nn.Sequential(
            nn.Linear(in_features=768, out_features=768 // 2),
            nn.ReLU(),
            nn.Linear(in_features=768 // 2, out_features=2)
        )

    def forward(self, x):
        features1 = x
        x = self.first_process(x)
        x = x.squeeze(1)
        features = x
        features = self.third_process(features)
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=-1)
        logits = self.second_process(features)
        return logits, features, features1

class DPGNet(nn.Module):
    def __init__(self, clip_model_name='ViT-L/14', pretrained='openai', pretrain_size=224, 
                 device='cuda', ddp=False,ln_tuning_enabled=True, clip_layer=None, total_steps=None):
        super().__init__()
        self.device = device
        self.ddp = ddp
        self.clip_layer = clip_layer or [23]
        self.ln_tuning_enabled = ln_tuning_enabled
        self.extract_layer = 23
        if pretrained == 'openai':
            self.clip_vit, _ = load(clip_model_name, device=device, download_root='/opt/data/private/your/')
        else:
            raise ValueError("Invalid pretrained option. Use 'openai'.")
        self.clip_vit.to(device)
        self.clip_vit.float()
        self.logit_scale = self.clip_vit.logit_scale
        self.embed_dim = self.clip_vit.visual.transformer.width
        self.num_block = len(self.clip_vit.visual.transformer.resblocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.use_mixup_for_cls = False
        self.prompt_learner = PromptLearner(
            clip_model=self.clip_vit,
            ctx_dim=768,
            language_length=20,
            language_depth=5,
            dtype=self.clip_vit.dtype,
            device=self.device
        )
        self.text_encoder = TextEncoder(self.clip_vit)
        self.classification_head = ClipFeatureHead(num_quires=256, embed_dim=1024, normalize_features=True)
        self.ln_post = self.clip_vit.visual.ln_post
        self.proj = self.clip_vit.visual.proj
        self.n_cls = 1
        self.K = 100
        self.dim = 768
        self.source_feat_bank_fake = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=device))
        self.target_feat_bank_fake = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=device))
        self.source_max_probs_fake = [0.0] * (self.n_cls * self.K)
        self.target_max_probs_fake = [0.0] * (self.n_cls * self.K)
        self.source_feat_bank_real = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=device))
        self.target_feat_bank_real = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=device))
        self.source_max_probs_real = [0.0] * (self.n_cls * self.K)
        self.target_max_probs_real = [0.0] * (self.n_cls * self.K)
        # self.confi = 0.7
        self.warm_up = 0
        self.real_weight = 2.0  # New: Weight factor for real samples
        self.confi = 0.9  # Initial threshold
        self.confi_min = 0.7  # Minimum threshold
        self.confi_max = 0.85  # Maximum threshold
        self.total_steps = None  # Total number of batches
        self.current_step = 0  # Current batch index
        self.dynamic_threshold_enabled = True  # Control dynamic threshold
        self.is_training = False

        self._apply_ln_tuning()
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Modified to not automatically average, facilitating weighting
        self.prob = []
        self.label = []
        self.features = []
        self.correct = 0
        self.total = 0
        # New: Pseudo-label statistics variables
        self.pseudo_correct = 0
        self.pseudo_incorrect = 0
        self.filtered_pseudo_correct = 0
        self.filtered_pseudo_incorrect = 0

        self.classification_head.apply(self._init_weights)
        self._freeze()
        self.to(device)

    def _apply_ln_tuning(self):
        if self.ln_tuning_enabled:
            peft_config = LNTuningConfig(target_modules=["ln_1", "ln_2", "ln_post", "ln_pre"])
            self.clip_vit.visual = get_peft_model(self.clip_vit.visual, peft_config)
            _logger.info("Applied LN-tuning to CLIP visual encoder, only layer normalization parameters are trainable.")

    def reset_step_counter(self):
        """Reset batch counter"""
        self.current_step = 0
        _logger.info("Step counter reset for dynamic threshold.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _freeze(self):
        for name, param in self.clip_vit.named_parameters():
            param.requires_grad = False
        if self.ln_tuning_enabled:
            for name, param in self.clip_vit.named_parameters():
                if "ln_1" in name or "ln_2" in name or "ln_post" in name or "ln_pre" in name:
                    param.requires_grad = True
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.prompt_learner.parameters():
            param.requires_grad = True

    def reset_feature_bank(self):
        self.source_max_probs_fake = [0.0] * (self.n_cls * self.K)
        self.target_max_probs_fake = [0.0] * (self.n_cls * self.K)
        self.source_max_probs_real = [0.0] * (self.n_cls * self.K)
        self.target_max_probs_real = [0.0] * (self.n_cls * self.K)
        self.source_feat_bank_fake = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=self.device))
        self.target_feat_bank_fake = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=self.device))
        self.source_feat_bank_real = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=self.device))
        self.target_feat_bank_real = nn.Parameter(torch.zeros(self.n_cls * self.K, self.dim, device=self.device))
        _logger.info("All feature banks reset.")

    def reset_pseudo_stats(self):
        """Reset pseudo-label statistics variables"""
        self.pseudo_correct = 0
        self.pseudo_incorrect = 0
        self.filtered_pseudo_correct = 0
        self.filtered_pseudo_incorrect = 0
        _logger.info("Pseudo label statistics reset.")

    @torch.no_grad()
    def construct_bank(self, train_loader_x, train_loader_u, epoch):
        self.eval()
        source_samples = 0
        target_samples = 0
        source_fake_updates = 0
        target_fake_updates = 0
        source_real_updates = 0
        target_real_updates = 0

        expected_bank_shape = (self.n_cls * self.K, self.dim)
        for bank_name, bank in [
            ("source_feat_bank_fake", self.source_feat_bank_fake),
            ("target_feat_bank_fake", self.target_feat_bank_fake),
            ("source_feat_bank_real", self.source_feat_bank_real),
            ("target_feat_bank_real", self.target_feat_bank_real)
        ]:
            if bank.shape != expected_bank_shape:
                _logger.warning(f"{bank_name} shape mismatch: expected {expected_bank_shape}, got {bank.shape}. Reinitializing.")
                self.reset_feature_bank()
        for probs_name, probs in [
            ("source_max_probs_fake", self.source_max_probs_fake),
            ("target_max_probs_fake", self.target_max_probs_fake),
            ("source_max_probs_real", self.source_max_probs_real),
            ("target_max_probs_real", self.target_max_probs_real)
        ]:
            if len(probs) != self.n_cls * self.K:
                _logger.warning(f"{probs_name} length mismatch: expected {self.n_cls * self.K}, got {len(probs)}. Reinitializing.")
                self.reset_feature_bank()

        for batch in train_loader_u:
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)
            target_samples += image.size(0)
            pred_dict = self.forward_u({"image": image, "label": label}, epoch, inference=False)
            features = pred_dict["features"]
            cls_logits = pred_dict["cls"]
            logits_test_u = pred_dict['logits_test_u']
            probs = torch.softmax(cls_logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask_fake = pseudo_labels == 1
            _logger.debug(f"Target batch with {mask_fake.sum().item()} fake samples")
            for i in torch.where(mask_fake)[0]:
                prob_i = max_probs[i].item()
                if prob_i > min(self.target_max_probs_fake) and prob_i < 0.9:
                    min_index = self.target_max_probs_fake.index(min(self.target_max_probs_fake))
                    self.target_max_probs_fake[min_index] = prob_i
                    self.target_feat_bank_fake[min_index] = features[i]
                    target_fake_updates += 1
            mask_real = pseudo_labels == 0
            _logger.debug(f"Target batch with {mask_real.sum().item()} real samples")
            for i in torch.where(mask_real)[0]:
                prob_i = max_probs[i].item()
                if prob_i > min(self.target_max_probs_real) and prob_i < 0.9:
                    min_index = self.target_max_probs_real.index(min(self.target_max_probs_real))
                    self.target_max_probs_real[min_index] = prob_i
                    self.target_feat_bank_real[min_index] = features[i]
                    target_real_updates += 1

        _logger.info(f"Feature bank constructed with {source_samples} source samples "
                     f"({source_fake_updates} fake updates, {source_real_updates} real updates) "
                     f"and {target_samples} target samples "
                     f"({target_fake_updates} fake updates, {target_real_updates} real updates).")

    def update_threshold(self, epoch, total_steps=None):
        """Update threshold based on epoch and batch progress"""
        if epoch == 1 and self.dynamic_threshold_enabled and self.is_training:
            if total_steps is not None:
                self.total_steps = total_steps
            if self.total_steps is None:
                _logger.warning("Total steps not set, cannot update threshold dynamically.")
                return
            progress = self.current_step / self.total_steps
            self.confi = self.confi_max - progress * (self.confi_max - self.confi_min)
            self.confi = max(self.confi_min, min(self.confi_max, self.confi))
            _logger.debug(f"Step {self.current_step}/{self.total_steps}, confi updated to {self.confi:.3f}")
            self.current_step += 1
        else:
            self.confi = 0.7  # Default threshold

    def forward(self, data_dict, epoch, inference=False):
        self.is_training = not inference
        images = data_dict['image'].to(self.device)
        labels = data_dict['label'].to(self.device)
        image_u = data_dict['image_u'].to(self.device) if 'image_u' in data_dict else None
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        clip_features = self.clip_vit.extract_features(clip_images, extract=self.clip_layer)
        clip_features = clip_features[self.extract_layer]

        # Cross-domain Mixup augmentation (only for source domain fake data)
        if not inference and image_u is not None and epoch > 0:
            clip_images_u = F.interpolate(image_u, size=(224, 224), mode='bilinear', align_corners=False)
            clip_features_u = self.clip_vit.extract_features(clip_images_u, extract=self.clip_layer)[self.extract_layer]  # [B_u, 256, 1024]

            mask_fake = labels == 1
            if mask_fake.sum() > 0:
                # Align batch sizes
                batch_fake = clip_features[mask_fake].size(0)
                batch_u = clip_features_u.size(0)
                if batch_u < batch_fake:
                    repeat_times = (batch_fake + batch_u - 1) // batch_u
                    clip_features_u = clip_features_u.repeat(repeat_times, 1, 1)[:batch_fake]
                elif batch_u > batch_fake:
                    indices = torch.randperm(batch_u, device=self.device)[:batch_fake]
                    clip_features_u = clip_features_u[indices]

                # Mixup augmentation (mimicking LSDA)
                alpha = torch.rand(1, device=self.device) * 1.5 + 0.5  # [0.5, 2]
                lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().to(self.device)
                cross_features = lambda_ * clip_features[mask_fake] + (1 - lambda_) * clip_features_u  # [B_fake, 256, 1024]

                # Save features for distillation
                self.features_orig_fake = clip_features[mask_fake].clone()
                self.cross_features = cross_features

                # Select classification features
                if self.use_mixup_for_cls:
                    fused_features = clip_features.clone()
                    fused_features[mask_fake] = cross_features  # Replace fake sample features
                else:
                    fused_features = clip_features  # Use only original features for classification
            else:
                fused_features = clip_features
                self.features_orig_fake = None
                self.cross_features = None
        else:
            fused_features = clip_features
            self.features_orig_fake = None
            self.cross_features = None

        cls_logits, features, features1 = self.classification_head(clip_features)
        prob = torch.softmax(cls_logits, dim=1)[:, 1]
        text = ['real face', 'fake face']
        prompts, tokenized_prompts = self.prompt_learner(self.clip_vit, text, self.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = torch.chunk(text_features, dim=0, chunks=2)
        text_features_mean = torch.stack([text_features[0].mean(0), text_features[1].mean(0)], dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)

        if not inference and epoch > 0:
            pred_dict = {
                'cls': cls_logits,
                'prob': prob,
                'features': features,
                'text_features': text_features_mean,
                'clip_features': clip_features
            } 
        elif inference and epoch > 0:
            logits_test_u = features @ text_features_mean.t()
            w_text = 0.5
            final_logits = (1 - w_text) * cls_logits + w_text * logits_test_u
            probs1 = torch.softmax(cls_logits, dim=1)[:, 1]
            probs2 = torch.softmax(logits_test_u, dim=1)[:, 1]
            probs = 0.5 * probs1 + 0.5 * probs2
            # pdb.set_trace()
            pred_dict = {
                'cls': final_logits,
                'prob': probs,
                'features': features,
                'text_features': text_features_mean,
                'clip_features': clip_features
            }
        elif inference and epoch == 0:
            pred_dict = {
                'cls': cls_logits,
                'prob': prob,
                'features': features,
                'text_features': text_features_mean,
                'clip_features': clip_features
            }
        else:
            pred_dict = {
                'cls': cls_logits,
                'prob': prob,
                'features': features,
                'text_features': text_features_mean,
                'clip_features': clip_features
            } 

        if inference:
            self.prob.append(prob.detach().cpu().numpy())
            self.label.append(labels.detach().cpu().numpy())
            self.features.append(features.detach().cpu().numpy())
            _, predicted = torch.max(cls_logits, 1)
            self.correct += (predicted == labels).sum().item()
            self.total += labels.size(0)

        return pred_dict

    def forward_u(self, data_dict, epoch, inference=False):
        images = data_dict['image'].to(self.device)
        labels = data_dict['label'].to(self.device)
        clip_images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        clip_features = self.clip_vit.extract_features(clip_images, extract=self.clip_layer)
        clip_features = clip_features[self.extract_layer]
        cls_logits, features, features1 = self.classification_head(clip_features)
        prob = torch.softmax(cls_logits, dim=1)[:, 1]
        text = ['an image', 'an image']
        prompts, tokenized_prompts = self.prompt_learner(self.clip_vit, text, self.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = torch.chunk(text_features, dim=0, chunks=2)
        text_features_mean = torch.stack([text_features[0].mean(0), text_features[1].mean(0)], dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
        logits_test_u = features @ text_features_mean.t()

        fake_bank_norm = self.target_feat_bank_fake  # [K, 768]
        real_bank_norm = self.target_feat_bank_real  # [K, 768]
        sim_to_fake = features @ fake_bank_norm.t()  # [B, K]
        sim_to_real = features @ real_bank_norm.t()  # [B, K]
        mean_sim_to_fake = sim_to_fake.max(dim=1)[0]   # [B]
        mean_sim_to_real = sim_to_real.mean(dim=1)  # [B]  
        # Count the number of correct and incorrect initial pseudo-labels
        probs = torch.softmax(cls_logits, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        correct_mask = pseudo_labels == labels
        self.pseudo_correct += correct_mask.sum().item()
        self.pseudo_incorrect += (pseudo_labels != labels).sum().item()
        
        # Count the number of correct and incorrect pseudo-labels after confidence filtering
        mask = max_probs.ge(self.confi).float()
        if mask.sum() > 0:
            filtered_correct_mask = (pseudo_labels == labels) & (mask == 1)
            self.filtered_pseudo_correct += filtered_correct_mask.sum().item()
            self.filtered_pseudo_incorrect += ((pseudo_labels != labels) & (mask == 1)).sum().item()
        
        pred_dict = {
            'cls': cls_logits,
            'prob': prob,
            'features': features,
            'text_features': text_features_mean,
            'logits_test_u': logits_test_u,
            'sim_to_fake': mean_sim_to_fake,
            'sim_to_real': mean_sim_to_real
        }
        if inference:
            self.prob.append(prob.detach().cpu().numpy())
            self.label.append(labels.detach().cpu().numpy())
            self.features.append(features.detach().cpu().numpy())
            _, predicted = torch.max(cls_logits, 1)
            self.correct += (predicted == labels).sum().item()
            self.total += labels.size(0)
        return pred_dict

    def compute_contrastive_loss(self, logits, pseudo_labels, mask=None, margin=1.0):
        real_logits = logits[:, 0]
        fake_logits = logits[:, 1]
        contrast_loss = torch.tensor(0.0, device=logits.device)
        if mask is not None:
            mask = mask.bool()
            mask_real = (pseudo_labels == 0) & mask
            mask_fake = (pseudo_labels == 1) & mask
        else:
            mask_real = pseudo_labels == 0
            mask_fake = pseudo_labels == 1

        if mask_real.sum() > 0:
            contrast_loss += torch.mean(F.relu(margin - (real_logits[mask_real] - fake_logits[mask_real])))
        if mask_fake.sum() > 0:
            contrast_loss += torch.mean(F.relu(margin - (fake_logits[mask_fake] - real_logits[mask_fake])))

        return contrast_loss

    def get_losses(self, data_dict, pred_dict, epoch=None, total_steps=None):
        self.update_threshold(epoch, total_steps)
        labels = data_dict['label'].to(self.device)
        cls_logits = pred_dict['cls']
        features = pred_dict['features']
        text_features = pred_dict['text_features']
        clip_features = pred_dict['clip_features']  # Use features from cls_logits as clip_features
        # Apply weighting for real samples
        weights = torch.ones_like(labels, dtype=torch.float, device=self.device)
        weights[labels == 0] = self.real_weight  # Weight real samples (label == 0)
        loss_cls = self.criterion(cls_logits, labels) * weights
        loss_cls = loss_cls.mean()

        probs = features @ text_features.t()
        loss_aux = self.criterion(probs, labels) * weights
        loss_aux = loss_aux.mean()
        
        loss_pseudo = torch.tensor(0.).to(self.device)
        loss_contrast_u = torch.tensor(0.0, device=self.device)
        loss_distill = torch.tensor(0.).to(self.device)
        if 'image_u' in data_dict and epoch > self.warm_up:
            image_u = data_dict['image_u'].to(self.device)
            label = data_dict['label_u'].to(self.device)
            pred_dict_u = self.forward_u({"image": image_u, "label": label}, epoch)
            logits_u = pred_dict_u['cls']
            features_u = pred_dict_u['features']
            text_features_u = pred_dict_u['text_features']
            probs_u_text = features_u @ text_features_u.t()
            probs_u = torch.softmax(logits_u, dim=1)
            max_probs, pseudo_labels = torch.max(probs_u, dim=1)
            sim_to_fake = pred_dict_u['sim_to_fake']
            
            # Refine pseudo-labels by combining confidence and similarity
            # Generate pseudo-labels based on similarity
            sim_pseudo_labels = (sim_to_fake > 0.5).long()  # 1 for fake, 0 for real
            # Keep only samples where similarity-based and confidence-based pseudo-labels match
            probs_u_text = features_u @ text_features_u.t()
            probs_u = torch.softmax(logits_u, dim=1)
            max_probs, pseudo_labels = torch.max(probs_u, dim=1)
            # loss_aux_u = self.criterion(probs_u_text, pseudo_labels)
            
            mask1 = max_probs.ge(self.confi).float()
            mask = (sim_pseudo_labels == pseudo_labels).float() * mask1
            if mask.sum() > 0:
                loss_pseudo = (self.criterion(probs_u_text, pseudo_labels) + self.criterion(logits_u, pseudo_labels)) * mask
                loss_contrast_u = self.compute_contrastive_loss(probs_u_text, pseudo_labels, mask=mask)
                loss_pseudo = loss_contrast_u + loss_pseudo.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.).to(self.device)
        mask_fake = labels == 1
        if mask_fake.sum() > 0 and self.cross_features is not None and epoch > 0:
            loss_distill = F.mse_loss(clip_features[mask_fake], self.cross_features)
        else:
            loss_distill = torch.tensor(0.0).to(self.device)
        if epoch > 0:
            loss_totall = 0.4 * loss_cls + 0.5 * loss_aux + loss_pseudo + 0.1 * loss_distill
        else:
            loss_totall = loss_cls + 0.8 * loss_aux + 0.5 * loss_pseudo 
        
        # Log pseudo-label statistics
        # _logger.info(f"Pseudo labels: correct={self.pseudo_correct}, incorrect={self.pseudo_incorrect}")
        # _logger.info(f"Filtered pseudo labels (confi={self.confi}): correct={self.filtered_pseudo_correct}, incorrect={self.filtered_pseudo_incorrect}")
        
        loss_dict = {
            'cls': loss_cls,
            'aux': loss_aux,
            'loss_pseudo': loss_pseudo,
            'loss_distill': loss_distill,
            'overall': loss_totall
        }
        return loss_dict

    def get_train_metrics(self, data_dict, pred_dict):
        labels = data_dict['label'].detach()
        cls_logits = pred_dict['cls'].detach()
        auc, eer, acc, ap = calculate_metrics_for_train(labels, cls_logits)
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        features = np.concatenate(self.features)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        ap = metrics.average_precision_score(y_true, y_pred)
        acc = self.correct / self.total if self.total > 0 else 0.0
        # Log and reset pseudo-label statistics
        _logger.info(f"Test pseudo labels: correct={self.pseudo_correct}, incorrect={self.pseudo_incorrect}")
        _logger.info(f"Test filtered pseudo labels (confi={self.confi}): correct={self.filtered_pseudo_correct}, incorrect={self.filtered_pseudo_incorrect}")
        self.reset_pseudo_stats()
        self.prob = []
        self.label = []
        self.features = []
        self.correct = 0
        self.total = 0
        return {
            'acc': acc,
            'auc': auc,
            'eer': eer,
            'ap': ap,
            'pred': y_pred,
            'label': y_true,
            'features': features
        }