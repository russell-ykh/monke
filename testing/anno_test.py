import pandas as pd

# # Sample Table B
# data_b = {'values': [3, 8, 12, 16, 21]}
# table_b = pd.DataFrame(data_b)

# # Sample Table A with boundaries and labels
# data_a = {'lower_bound': [0, 5, 10, 15, 20],
#           'upper_bound': [4, 9, 14, 19, 25],
#           'label': ['A', 'B', 'C', 'D', 'E']}
# table_a = pd.DataFrame(data_a)

# # Merge the two tables based on the boundaries
# merged_table = pd.merge_asof(table_b, table_a)

# # Drop unnecessary columns
# merged_table = merged_table.drop(['lower_bound', 'upper_bound'], axis=1)

# # Display the merged table
# print(merged_table)