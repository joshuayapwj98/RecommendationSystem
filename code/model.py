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
        # Prediction score for the user-item pair
        # First order - Linear regression
        # Second order - Dot product (Pairwise interaction)
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean)
    
# TODO: Create new model class
# Must have: __init__, forward functions
class ExtendedMF(MF):
    def __init__(self, num_users, num_items, embedding_size=8, dropout=0, mean=0, category_dict = None, visual_dict = None):
        super(ExtendedMF, self).__init__(num_users, num_items, embedding_size, dropout, mean)
        
        assert category_dict is not None and visual_dict is not None # dictionaries must not be empty

        self.category_dict = category_dict
        self.visual_dict = visual_dict

        # Embedding layers for categorical features
        self.category_emb = nn.Embedding(num_items, embedding_size)
       
        # Linear layer for visual features
        self.visual_linear = nn.Linear(512, embedding_size)
        
        # Initialize weights
        self.category_emb.weight.data.uniform_(0, 0.005)
        self.visual_linear.weight.data.uniform_(0, 0.005)

    def forward(self, u_id, i_id, category, visual):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        
        # Get category and visual features for the item
        category_feature = self.category_emb(category.long())
        visual_feature = self.visual_linear(visual.float())

        # Combine item embedding with category and visual features
        I_combined = I + category_feature + visual_feature
        
        # Prediction score for the user-item pair
        # First order - Linear regression
        # Second order - Dot product (Pairwise interaction)
        return self.dropout((U * I_combined).sum(1) + b_u + b_i + self.mean)