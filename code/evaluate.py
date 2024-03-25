import numpy as np

import torch

from model import ExtendedMF

# TODO: design a metric to evaluate the model
# Write a evalution for the diversity
def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, category_dict, visual_dict):
	recommends = []
	for i in range(len(top_k)):
		recommends.append([])

	with torch.no_grad():
		category_array, visual_array = None, None
		if isinstance(model, ExtendedMF):
			category_array = np.array([category_dict[i] for i in range(len(category_dict))])
			category_tensor = torch.tensor(category_array).to(args.device)

			visual_array = np.array([visual_dict[i] for i in range(len(visual_dict))])
			visual_tensor = torch.tensor(visual_array).to(args.device)

		pred_list_all = []
		for i in gt_dict.keys(): # for each user
			if len(gt_dict[i]) != 0: # if 
				user = torch.full((item_num,), i, dtype=torch.int64).to(args.device) # create n_item users for prediction
				item = torch.arange(0, item_num, dtype=torch.int64).to(args.device) 

				if isinstance(model, ExtendedMF):
					category = category_tensor[item]
					visual = visual_tensor[item]
					prediction = model(user, item, category, visual)
				else:
					prediction = model(user, item)
					
				prediction = prediction.detach().cpu().numpy().tolist()
				for j in train_dict[i]: # mask train
					prediction[j] -= float('inf')
				if flag == 1: # mask validation
					if i in valid_dict:
						for j in valid_dict[i]:
							prediction[j] -= float('inf')
				pred_list_all.append(prediction)

		predictions = torch.Tensor(pred_list_all).to(args.device) # shape: (n_user,n_item)
		for idx in range(len(top_k)):
			_, indices = torch.topk(predictions, int(top_k[idx]))
			recommends[idx].extend(indices.tolist())
	return recommends

# TODO: implement another coverage for category instead of item
# 
def catalog_coverage(recommends, item_num):
    """
    Calculate catalog coverage.

    Args:
        recommends (list): List of recommended items for each user.
        item_num (int): Total number of items in the catalog.

    Returns:
        float: Catalog coverage.
    """
    recommended_items = set()
    for user_recommendations in recommends:
        for item_list in user_recommendations:
            for item in item_list:
                recommended_items.add(item)
    coverage = len(recommended_items) / item_num
    return coverage

def average_similarity(recommends, item_embeddings):
    """
    Calculate the average pairwise cosine similarity between recommended items.

    Args:
        recommends (list): List of recommended items for each user.
        item_embeddings (torch.Tensor): Embeddings of all items.

    Returns:
        float: Average pairwise cosine similarity.
    """
    similarities = []
    for user_recommendations in recommends:
        for item_list in user_recommendations:
            # Convert indices to embeddings
            embeddings = item_embeddings[item_list]
            # Calculate pairwise cosine similarity
            similarity_matrix = torch.matmul(embeddings, embeddings.t())
            norms = torch.norm(embeddings, dim=1)
            similarity_matrix /= torch.outer(norms, norms)
            similarity_matrix = similarity_matrix.fill_diagonal_(0)
            # Average similarity for the item list
            average_similarity = similarity_matrix.sum() / (len(item_list) * (len(item_list) - 1))
            similarities.append(average_similarity.item())
    # Calculate the overall average similarity
    if similarities:
        return sum(similarities) / len(similarities)
    else:
        return 0.0
    
def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, category_dict, visual_dict):
	RECALL, NDCG = [], []
	recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, category_dict, visual_dict)
	# TODO: return a list of recommendation lists for each user
	# [[1,2,4,6,4,3,10,21,26,55], [777,633,12,53,25,123,5,234,17,15]]
	# Where each list represents the recommended item for each user
	coverage = catalog_coverage(recommends, item_num)

	# Get item embeddings for calculating similarity
	item_embeddings = model.item_emb.weight

	# Calculate average similarity
	avg_similarity = average_similarity(recommends, item_embeddings)
		
	for idx in range(len(top_k)):
		sumForRecall, sumForNDCG, user_length = 0, 0, 0
		k=-1
		for i in gt_dict.keys(): # for each user
			k += 1
			if len(gt_dict[i]) != 0:
				userhit = 0
				dcg = 0
				idcg = 0
				idcgCount = len(gt_dict[i])
				ndcg = 0

				for index, thing in enumerate(recommends[idx][k]):
					if thing in gt_dict[i]:
						userhit += 1
						dcg += 1.0 / (np.log2(index+2))
					if idcgCount > 0:
						idcg += 1.0 / (np.log2(index+2))
						idcgCount -= 1
				if (idcg != 0):
					ndcg += (dcg / idcg)

				sumForRecall += userhit / len(gt_dict[i])
				sumForNDCG += ndcg
				user_length += 1

		RECALL.append(round(sumForRecall/user_length, 4))
		NDCG.append(round(sumForNDCG/user_length, 4))

	return RECALL, NDCG, coverage, avg_similarity

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        if len(valid_result) >= 4:
            print("[Valid]: Recall: {} NDCG: {} Coverage: {:.4f} Avg. Similarity: {:.4f}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]),
                                valid_result[2], valid_result[3]))
        else:
            print("[Valid]: Recall: {} NDCG: {}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]])))
    if test_result is not None: 
        if len(test_result) >= 4:
            print("[Test]: Recall: {} NDCG: {} Coverage: {:.4f} Avg. Similarity: {:.4f}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]),
                                test_result[2], test_result[3]))
        else:
            print("[Test]: Recall: {} NDCG: {}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]])))
            
            # TODO: Get the final recommendation list, calculate recall value and diversity from the test
            # TODO: plot the f1 measure of the recall and diversity
             