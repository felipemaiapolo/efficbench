import numpy as np
import random
from irt import *
from utils import *

def get_seen_unseen_items(scenarios_choosen, scenarios, number_item, subscenarios_position, responses_test):
    
    """
    Stratified sample items (seen_items). 'unseen_intems' gives the complement.
    
    Parameters:
    - scenarios_choosen: A list of considered scenarios.
    - scenarios: A dictionary where keys are scenario identifiers and values are lists of subscenarios.
    - number_item: The total number of items to be considered across all chosen scenarios.
    - subscenarios_position: A nested dictionary where the first key is the scenario and the second key is the subscenario, 
      and the value is a list of item positions for that subscenario.
    - responses_test: A numpy array of the test subject's responses to all items. (this is only used to get the number of items)
    
    Returns:
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    """
    
    def shuffle_list(lista):
        """
        Shuffles a list in place and returns the shuffled list.
        
        Parameters:
        - lista: The list to be shuffled.
        
        Returns:
        - A shuffled version of the input list.
        """
        return random.sample(lista, len(lista))

    
    seen_items = []  # Initialize an empty list to hold the indices of seen items.
    # Iterate through each chosen scenario to determine the seen items.
    for scenario in scenarios_choosen:
        # Allocate the number of items to be seen in each subscenario.
        number_items_sub = np.zeros(len(scenarios[scenario])).astype(int)
        number_items_sub += number_item // len(scenarios[scenario])
        number_items_sub[:(number_item - number_items_sub.sum())] += 1
        
        i = 0  # Initialize a counter for the subscenarios.
        # Shuffle the subscenarios and iterate through them to select seen items.
        for sub in shuffle_list(scenarios[scenario]):
            # Randomly select items from the subscenario and add them to the seen items.
            seen_items += random.sample(subscenarios_position[scenario][sub], k=number_items_sub[i])
            i += 1

    # Determine the unseen items by finding all item indices that are not in the seen items list.
    unseen_items = [i for i in range(responses_test.shape[1]) if i not in seen_items]

    return seen_items, unseen_items
