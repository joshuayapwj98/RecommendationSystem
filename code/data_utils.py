import numpy as np
import torch.utils.data as data

# TODO: load additional categorical features here


def load_all(train_path, valid_path, test_path, category_path=None, visual_path=None):
    """ We load all the three file here to save time in each epoch. """
    train_dict = np.load(train_path, allow_pickle=True).item()
    valid_dict = np.load(valid_path, allow_pickle=True).item()
    test_dict = np.load(test_path, allow_pickle=True).item()
    
    category_dict = visual_dict = None

    if category_path is not None and visual_path is not None:
        category_dict = np.load(category_path, allow_pickle=True).item()
        visual_dict = np.load(visual_path, allow_pickle=True).item()

    user_num, item_num = 0, 0
    user_num = max(user_num, max(train_dict.keys()))
    user_num = max(user_num, max(valid_dict.keys()))
    user_num = max(user_num, max(test_dict.keys()))

    train_data, valid_gt, test_gt = [], [], []
    for user, items in train_dict.items():
        item_num = max(item_num, max(items))
        for item in items:
            train_data.append([int(user), int(item)])
            # if category_dict is not None and visual_dict is not None:
            #     train_data[-1].extend([category_dict[item], visual_dict[item]])
    for user, items in valid_dict.items():
        item_num = max(item_num, max(items))
        for item in items:
            valid_gt.append([int(user), int(item)])
            # if category_dict is not None and visual_dict is not None:
            #     valid_gt[-1].extend([category_dict[item], visual_dict[item]])
    for user, items in test_dict.items():
        item_num = max(item_num, max(items))
        for item in items:
            test_gt.append([int(user), int(item)])
            # if category_dict is not None and visual_dict is not None:
            #     test_gt[-1].extend([category_dict[item], visual_dict[item]])

    return user_num+1, item_num+1, train_dict, valid_dict, test_dict, category_dict, visual_dict, train_data, valid_gt, test_gt


class MFData(data.Dataset):
    def __init__(self, features, num_item, train_dict=None, is_training=None, is_extended=False, category_dict=None, visual_dict=None):
        super(MFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
        self.features_ps = features
        self.num_item = num_item
        self.train_dict = train_dict
        self.category_dict = category_dict
        self.visual_dict = visual_dict
        self.is_training = is_training
        self.is_extended= is_extended
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to do sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            j = np.random.randint(self.num_item)
            while j in self.train_dict[u]:
                j = np.random.randint(self.num_item)
            self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        """ Override the __len__ method
        """
        return (1 + 1) * len(self.labels)

    # TODO: return additional categorical features here [DONE]
    def __getitem__(self, idx):
        """ Override the __getitem__ method
        """
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels

        category, visual = None, []

        user = features[idx][0]
        item = features[idx][1]
        if self.is_extended:
            category = self.category_dict[item]
            visual = self.visual_dict[item]
            
        label = labels[idx]
        
        if self.is_extended:
            return user, item, label, category, visual
        else:
            return user, item, label
