# WeightLoss-Network-Analysis
*Original data stored in* `WeightLoss` *and* `WeightLoss/networks` *folders*

## Data Pre-processing
*All processed data stored in* `ProcessedData` *folder*

`Consolidate_Data_and_Filter_Missing.ipynb` - consolidates all useful data from original and stores in `valid_user_info_consolidated.csv`

`Network_Filtering.ipynb` - detects subset of user nodes that have degrees of at least 5 in all 3 subgraphs. Stores nodes in `included_users_(sorted).csv` and subgraph edge lists in `friend_edges.csv`, `comment_edges.csv`, and `mention_edges.csv`

`Label_and_Feature_Extraction.ipynb` - filters consolidated data in `valid_user_info_consolidated.csv` by users in `included_users_(sorted).csv`. Then splits into features and labels, stored in `features.csv` and `labels.csv` respectively

## Models
`Model Training.ipynb` - not implemented yet
