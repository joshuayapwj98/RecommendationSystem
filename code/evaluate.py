import numpy as np

import torch

from model import ExtendedMF
import json

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

def calculate_ild(recommendations, category_dict, top_k):
    """
    Calculate Intra-List-Diversity (ILD) for each user based on item categories and the average ILD.
    
    Args:
        recommendations (list): A list of lists of lists where each inner list contains the recommended item IDs for a user for a specific value of top_k.
        category_dict (dict): A dictionary where keys are item IDs and values are the categories of the items.
        top_k (list): A list of the number of top recommendations to consider.
        
    Returns:
        list: A list of lists of ILD scores for each user for each value of top_k.
        list: A list of the average ILD score for each value of top_k.
    """
    ild_scores = [[] for _ in range(len(top_k))]
    avg_ild = [0 for _ in range(len(top_k))]
    num_users = len(recommendations[0])
    for idx, k in enumerate(top_k):
        total_ild = 0
        for user in range(num_users):
            recommended_items = recommendations[idx][user]
            categories = np.array([category_dict[item] for item in recommended_items])
            category_matrix = np.repeat(categories, k).reshape(k, k)
            ild = np.sum(category_matrix != category_matrix.T) / 2
            ild /= (k * (k - 1) / 2)
            ild_scores[idx].append(ild)
            total_ild += ild
        avg_ild[idx] = round(total_ild / num_users, 4)
    return ild_scores, avg_ild

def calculate_f1(ndcg_scores, ild_scores, top_k):
    """
    Calculate F1 measure (NDCG-ILD) for each user and the average F1.
    
    Args:
        ndcg_scores (list): A list of lists of NDCG scores for each user for each value of top_k.
        ild_scores (list): A list of lists of ILD scores for each user for each value of top_k.
        top_k (list): A list of the number of top recommendations to consider.
        
    Returns:
        list: A list of lists of F1 scores for each user for each value of top_k.
        list: A list of the average F1 score for each value of top_k.
    """
    f1_scores = [[] for _ in range(len(top_k))]
    avg_f1 = [0 for _ in range(len(top_k))]
    num_users = len(ndcg_scores[0])
    for idx, _ in enumerate(top_k):
        total_f1 = 0
        for user in range(num_users):
            ndcg = ndcg_scores[idx][user]
            ild = ild_scores[idx][user]
            f1 = (2 * ndcg * ild) / (ndcg + ild) if (ndcg + ild) != 0 else 0
            f1_scores[idx].append(f1)
            total_f1 += f1
        avg_f1[idx] = round(total_f1 / num_users, 4)
    return f1_scores, avg_f1

def item_coverage(recommends, item_num):
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

def calculate_category_coverage(recommendations, category_dict, category_set):
    """
    Calculate the category coverage of the recommendations.

    Args:
        recommendations (list): List of recommended items for each user.
        category_set (set): Set of all category IDs.

    Returns:
        float: Category coverage.
    """
    recommended_categories = set()
    for item_recommendations in recommendations:
        category = category_dict[item_recommendations]
        recommended_categories.add(category)
    # Look at the coverage function again
    coverage = (len(recommended_categories) / len(category_set)) * 100
    return coverage
    
def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, category_dict, visual_dict, categories):
    RECALL, AVG_NDCG, NDCG = [], [], [[] for _ in range(len(top_k))]
    recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, category_dict, visual_dict)

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
                NDCG[idx].append(ndcg)

        RECALL.append(round(sumForRecall/user_length, 4))
        AVG_NDCG.append(round(sumForNDCG/user_length, 4))

    ild_scores, avg_ild = calculate_ild(recommends, category_dict, top_k)
    _, avg_f1 = calculate_f1(NDCG, ild_scores, top_k)
    
    return RECALL, AVG_NDCG, avg_ild, avg_f1, recommends

def write_recommendations(recommends, category_dict, category_set, output_file):
    """
    Write the recommendations to a JSON file.

    Args:
        recommends (list): List of recommended items for each user.
        output_file (str): Path to the output JSON file.
    """
    data = []
    for user_id, item_recommendations in enumerate(recommends):
        user_data = {
            "user_id": user_id,
            "recommendations": item_recommendations,
            "category_coverage": calculate_category_coverage(item_recommendations, category_dict, category_set)
        }
        data.append(user_data)
    
    with open(output_file, 'w') as f:
        json.dump(data, f)

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Recall: {} NDCG: {} ILD: {} F1: {} ".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]),
                            '-'.join([str(x) for x in valid_result[2]]),
                            '-'.join([str(x) for x in valid_result[3]]),
                            ))
    if test_result is not None: 
        print("[Test]: Recall: {} NDCG: {} ILD: {} F1: {} ".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]),
                            '-'.join([str(x) for x in test_result[2]]),
                            '-'.join([str(x) for x in test_result[3]]),
                            ))
             
             