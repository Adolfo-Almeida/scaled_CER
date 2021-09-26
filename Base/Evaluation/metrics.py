import numpy as np


class _Metrics_Object(object):
    """
    Abstract class that should be used as superclass of all metrics requiring an object, therefore a state, to be computed
    """
    def __init__(self):
        pass

    def __str__(self):
        return "{:.4f}".format(self.get_metric_value())

    def add_recommendations(self, recommended_items_ids):
        raise NotImplementedError()

    def get_metric_value(self):
        raise NotImplementedError()

    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()


####################################################################################################################
###############                 ACCURACY METRICS
####################################################################################################################


class MAP(_Metrics_Object):
    """
    Mean Average Precision, defined as the mean of the AveragePrecision over all users

    """

    def __init__(self):
        super(MAP, self).__init__()
        self.cumulative_AP = 0.0
        self.n_users = 0

    def add_recommendations(self, is_relevant, pos_items):
        self.cumulative_AP += average_precision(is_relevant, pos_items)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_AP/self.n_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is MAP, "MAP: attempting to merge with a metric object of different type"

        self.cumulative_AP += other_metric_object.cumulative_AP
        self.n_users += other_metric_object.n_users



def average_precision(is_relevant, pos_items):

    if len(is_relevant) == 0:
        a_p = 0.0
    else:
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
        a_p = np.sum(p_at_k) / np.min([pos_items.shape[0], is_relevant.shape[0]])

    assert 0 <= a_p <= 1, a_p
    return a_p



def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    # Creates array of length "at" with the relevance associated to the item in that position
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)

    # IDCG has all relevances to 1, up to the number of items in the test set
    ideal_dcg = dcg(np.sort(relevance)[::-1])

    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(rank_scores)

    if rank_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)





####################################################################################################################
###############                 BEYOND-ACCURACY METRICS
####################################################################################################################


class _Global_Item_Distribution_Counter(_Metrics_Object):
    """
    This abstract class implements the basic functions to calculate the global distribution of items
    recommended by the algorithm and is used by various diversity metrics
    """
    def __init__(self, n_items, ignore_items):
        super(_Global_Item_Distribution_Counter, self).__init__()

        self.recommended_counter = np.zeros(n_items, dtype=np.float)
        self.ignore_items = ignore_items.astype(np.int).copy()


    def add_recommendations(self, recommended_items_ids):
        if len(recommended_items_ids) > 0:
            self.recommended_counter[recommended_items_ids] += 1

    def _get_recommended_items_counter(self):

        recommended_counter = self.recommended_counter.copy()

        recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
        recommended_counter_mask[self.ignore_items] = False

        recommended_counter = recommended_counter[recommended_counter_mask]

        return recommended_counter


    def merge_with_other(self, other_metric_object):
        assert isinstance(other_metric_object, self.__class__), "{}: attempting to merge with a metric object of different type".format(self.__class__)

        self.recommended_counter += other_metric_object.recommended_counter

    def get_metric_value(self):
        raise NotImplementedError()




class Coverage_Item(_Global_Item_Distribution_Counter):
    """
    Item coverage represents the percentage of the overall items which were recommended
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff
    """

    def __init__(self, n_items, ignore_items):
        super(Coverage_Item, self).__init__( n_items, ignore_items)

    def get_metric_value(self):

        recommended_mask = self._get_recommended_items_counter() > 0

        return recommended_mask.sum()/len(recommended_mask)



class Shannon_Entropy(_Global_Item_Distribution_Counter):
    """
    Shannon Entropy is a well known metric to measure the amount of information of a certain string of data.
    Here is applied to the global number of times an item has been recommended.

    It has a lower bound and can reach values over 12.0 for random recommenders.
    A high entropy means that the distribution is random uniform across all users.

    Note that while a random uniform distribution
    (hence all items with SIMILAR number of occurrences)
    will be highly diverse and have high entropy, a perfectly uniform distribution
    (hence all items with EXACTLY IDENTICAL number of occurrences)
    will have 0.0 entropy while being the most diverse possible.

    """

    def __init__(self, n_items, ignore_items):
        super(Shannon_Entropy, self).__init__(n_items, ignore_items)

    def get_metric_value(self):

        recommended_counter = self._get_recommended_items_counter()
        shannon_entropy = _compute_shannon_entropy(recommended_counter)

        return shannon_entropy





def _compute_shannon_entropy(recommended_counter):

    # Ignore from the computation both ignored items and items with zero occurrence.
    # Zero occurrence items will have zero probability and will not change the result, butt will generate nans if used in the log
    recommended_counter_mask = np.ones_like(recommended_counter, dtype = np.bool)
    recommended_counter_mask[recommended_counter == 0] = False

    recommended_counter = recommended_counter[recommended_counter_mask]

    n_recommendations = recommended_counter.sum()

    recommended_probability = recommended_counter/n_recommendations

    shannon_entropy = -np.sum(recommended_probability * np.log2(recommended_probability))

    return shannon_entropy





class Diversity_similarity(_Metrics_Object):
    """
    Intra list diversity computes the diversity of items appearing in the recommendations received by each single user, by using an item_diversity_matrix.

    It can be used, for example, to compute the diversity in terms of features for a collaborative recommender.

    A content-based recommender will have low IntraList diversity if that is computed on the same features the recommender uses.
    A TopPopular recommender may exhibit high IntraList diversity.

    """

    def __init__(self, item_diversity_matrix):
        super(Diversity_similarity, self).__init__()

        assert np.all(item_diversity_matrix >= 0.0) and np.all(item_diversity_matrix <= 1.0), \
            "item_diversity_matrix contains value greated than 1.0 or lower than 0.0"

        self.item_diversity_matrix = 1.0 - item_diversity_matrix.astype(np.float32)

        self.n_evaluated_users = 0
        self.diversity = 0.0


    def add_recommendations(self, recommended_items_ids):

        current_recommended_items_diversity = 0.0

        for item_index in range(len(recommended_items_ids)-1):

            item_id = recommended_items_ids[item_index]

            item_other_diversity = self.item_diversity_matrix[item_id, recommended_items_ids]
           

            item_other_diversity[item_index] = 0.0

            current_recommended_items_diversity += np.sum(item_other_diversity)


        self.diversity += current_recommended_items_diversity/(len(recommended_items_ids)*(len(recommended_items_ids)-1))

        self.n_evaluated_users += 1


    def get_metric_value(self):

        if self.n_evaluated_users == 0:
            return 0.0

        return self.diversity/self.n_evaluated_users

    def merge_with_other(self, other_metric_object):
        assert other_metric_object is Diversity_similarity, "Diversity: attempting to merge with a metric object of different type"

        self.diversity = self.diversity + other_metric_object.diversity
        self.n_evaluated_users = self.n_evaluated_users + other_metric_object.n_evaluated_users