# FUNCTIONS FOR DATASET

def get_nodes(data):
    merged_accounts = pd.concat([data['Account'], data['Account.1']])
    merged_banks = pd.concat([data['From Bank'], data['To Bank']])
    merged_currencies = pd.concat([data['Receiving Currency'], data['Payment Currency']])
    merged_df = pd.DataFrame({
        'Accounts': merged_accounts,
        'Bank': merged_banks,
        'Currency': merged_currencies
    })
    unique_accounts = merged_df.drop_duplicates(subset=['Accounts']).reset_index(drop=True)
    return unique_accounts

def one_hot_encoding(unique_accounts, column):
    # Convert non-numeric columns
    positions = unique_accounts[column].str.split(",", expand=True) # creating new columns by splitting receiving currency --> all are added
    unique_accounts["first_position"] = positions[0] # first currency in each row is extracted --> actual currency used and that we want as TRUE
    
    # One-hot encoding
    table = pd.concat([unique_accounts, pd.get_dummies(unique_accounts["first_position"],dtype='int')], axis=1, join='inner') # effectively adds actual currency to dummy variables/columns
    table.drop([column, "first_position"], axis=1, inplace=True) # drop the axiliary columns
    return table

def normalize(table, new_min=0, new_max=10):
    # Check if the input table has only one column
    if len(table.columns) == 1:
        normalized_df = ((table - table.min()) / (table.max() - table.min())) * (new_max - new_min) + new_min
        return normalized_df
    else:
        normalized_df = pd.DataFrame()
        column_index = 0
        # Iterate through each column in the table
        for column in table.columns:
            column_data = table[column]
            if column_index == 0:
                # Normalize the first column
                normalized_df[f'col_{column_index}'] = ((column_data - column_data.min()) / (column_data.max() - column_data.min())) * (new_max - new_min) + new_min
            else:
                # Normalize subsequent columns
                normalized_column = ((column_data - column_data.min()) / (column_data.max() - column_data.min())) * (new_max - new_min) + new_min
                normalized_df[f'col_{column_index}'] = normalized_column
            column_index += 1
        return normalized_df


def hashing_vectorization(strings, vector_size=9):
    vectors = []
    for string in strings:
        # Hash the string using hash()
        hashed_values = hash(string) % (10 ** vector_size)  # Ensures unique representations within the specified range
        # Convert hashed values to a fixed-size vector
        vector = [int(digit) for digit in str(hashed_values)]
        # Ensure vector has the desired size by zero-padding or truncating
        if len(vector) < vector_size:
            vector = [0] * (vector_size - len(vector)) + vector
        elif len(vector) > vector_size:
            vector = vector[:vector_size]
        vectors.append(vector)
    return vectors

def get_longest_string_in_list(lst):
    res = max(lst, key=len)
    res_len = len(res)
    print(f"Longest String is  - {res} w/ length - {res_len}")
    return res, res_len

def make_binary_fixed_length(binary_lists, res):
    new_binary_list = []
    for x in binary_lists:
        if len(x) < len(res):
            num_zeros = len(res) - len(x)
            x = [0] * num_zeros + x
            new_binary_list.append(x)
        else:
            new_binary_list.append(x)
    return new_binary_list

def count_unused_decimals(number):
    # Convert number to string to iterate through digits
    num_str = str(number)
    count = 0

    # Iterate through digits from the end
    for digit in reversed(num_str):
        # If the digit is '0', increment count
        if digit == '0':
            count += 1
        # If non-zero digit encountered, break the loop
        else:
            break

    # Remove trailing zeroes from the number
    num_str = num_str.rstrip('0')

    return num_str, count

def split_into_vectors(table):
    lists = []
    for binary in table:
        binary = float(binary)
        binary = str(format(binary, 'f'))
        decimal_repr = []
        for bit in binary:
            if '.' not in bit:
                decimal_repr.append(str(int(bit)))
            else:
                decimal_repr.append(bit)
        lists.append(decimal_repr)
    return lists

def encode_payment_amount(df_col, max_len, min_len):
    new_payment_list = []
    for x in df_col:
        # print(x)
        index_of_decimal = x.index('.')
        positive_decimals = index_of_decimal
        negative_decimals = len(x) - (index_of_decimal+1)
        if positive_decimals < max_len:
            num_zeros = max_len - positive_decimals
            x = ['0'] * num_zeros + x
            # print(x)
            new_payment_list.append(x)
        elif negative_decimals < min_len:
            num_zeros = max_len - negative_decimals
            x = ['0'] * num_zeros + x
            # print(x)
            new_payment_list.append(x)
        else:
            new_payment_list.append(x)
        x.remove('.')
    return new_payment_list

def create_graph(edge_connections, edges_amount, limit=None):
    graph = nx.Graph()
    if limit is None:
        for i in range(len(edge_connections)):
            u = edge_connections[i].get("source")
            v = edge_connections[i].get("destination")
            graph.add_edge(u,v,label=edges_amount[i])
    else:
        for i in range(0, limit):
            u = edge_connections[i].get("source")
            v = edge_connections[i].get("destination")
            graph.add_edge(u,v,label=edges_amount[i])
    print(graph.edges(data=True))
    return graph