import torch
import torch.nn as nn
    
class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=8, dropout=0, mean=0):
        super(MF, self).__init__()
        # Embedding layers for users
        self.user_emb = nn.Embedding(num_users, embedding_size)
        # Bias terms to add to the user side
        self.user_bias = nn.Embedding(num_users, 1)
        # Embedding layers for items
        self.item_emb = nn.Embedding(num_items, embedding_size)
        # Bias terms to add to the item side
        self.item_bias = nn.Embedding(num_items, 1)

        # Uniform initalization for learning parameters
        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # Global bias term (learnable)
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        # Random dropout layer, to improve robustness of the model
        self.dropout = nn.Dropout(dropout)

        self.num_user = num_users

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean)
    
class ExtendedMF(MF):
    def __init__(self, num_users, num_items, embedding_size=8, dropout=0.5, mean=0, category_dict = None, visual_dict = None):
        super(ExtendedMF, self).__init__(num_users, num_items, embedding_size, dropout, mean)
        
        assert category_dict is not None and visual_dict is not None # dictionaries must not be empty

        self.category_dict = category_dict
        self.visual_dict = visual_dict

        # Embedding layers for categorical features
        self.category_emb = nn.Embedding(num_items, embedding_size)
       
        # Linear layer for visual features
        self.visual_linear = nn.Linear(512, embedding_size)
        
        # # Initialize weights with uniform distribution
        # self.category_emb.weight.data.uniform_(0, 0.005)
        # self.visual_linear.weight.data.uniform_(0, 0.005)
        
        # Initialize weights with xavier_uniform distribution
        nn.init.xavier_uniform_(self.category_emb.weight)
        nn.init.xavier_uniform_(self.visual_linear.weight)
        
        # Add a FFNN for feature interaction
        self.feature_interaction = nn.Sequential(
            nn.Linear(embedding_size * 3, embedding_size * 2),
            nn.BatchNorm1d(embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, embedding_size)
        )
    
    def forward(self, u_id, i_id, category, visual):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        
        # Get category and visual features for the item, normalize them
        category_feature = self.category_emb(category.long())
        category_feature = (category_feature - category_feature.mean()) / category_feature.std()
        
        visual_feature = self.visual_linear(visual.float())
        visual_feature = (visual_feature - visual_feature.mean()) / visual_feature.std()

        features = torch.cat([I, category_feature, visual_feature], dim=1)
        I_combined = self.feature_interaction(features)
        
        return self.dropout((U * I_combined).sum(1) + b_u + b_i + self.mean)
    
    # def calculate_diversity_loss(self, user, item, prediction):
    #     unique_users = torch.unique(user) 
    #     diversity_loss = 0.0

    #     for u in unique_users:
    #         user_item_indices = item[user == u]
    #         user_predictions = prediction[user == u]

    #         item_embeddings = self.item_emb(user_item_indices)
    #         prediction_products = torch.outer(user_predictions, user_predictions)

    #         # Compute pairwise dot products of item embeddings
    #         dot_products = item_embeddings @ item_embeddings.t()

    #         # Compute diversity loss for this user
    #         upper_triangle_indices = torch.triu_indices(len(user_item_indices), len(user_item_indices), offset=1)
    #         user_diversity_loss = (dot_products[upper_triangle_indices] / prediction_products[upper_triangle_indices]).sum()

    #         diversity_loss += user_diversity_loss

    #     diversity_loss /= len(unique_users)  # Average over all users
    #     return diversity_loss

class FM(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embedding_size):
        super(FM, self).__init__()

        # Embedding layers for users, items, and categories
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.category_emb = nn.Embedding(num_categories, embedding_size)
        self.visual_linear = nn.Linear(512, embedding_size)

        # Linear layers for first-order term
        self.user_linear = nn.Linear(1, 1)
        self.item_linear = nn.Linear(1, 1)
        self.category_linear = nn.Linear(1, 1)
        self.visual_linear_1st = nn.Linear(512, 1)

        # Bias term for first-order term
        self.bias = nn.Parameter(torch.zeros(1))

        # Initialize weights with xavier_uniform distribution
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.category_emb.weight)
        nn.init.xavier_uniform_(self.visual_linear.weight)

    def forward(self, user, item, category, visual):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        category_emb = self.category_emb(category)
        visual_emb = self.visual_linear(visual)

        # Compute the pairwise interactions using FM
        pairwise_interactions = torch.sum((user_emb * item_emb * category_emb * visual_emb), dim=1)

        # Compute the first-order term
        user_linear = self.user_linear(user.float().unsqueeze(1))
        item_linear = self.item_linear(item.float().unsqueeze(1))
        category_linear = self.category_linear(category.float().unsqueeze(1))
        visual_linear_1st = self.visual_linear_1st(visual)
        first_order = self.bias + user_linear + item_linear + category_linear + visual_linear_1st

        return first_order + pairwise_interactions