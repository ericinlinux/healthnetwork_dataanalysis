import os
import pandas as pd
import pickle
import xmltodict


def getNodesDegreeAtDate(date, nodes, edges):
    '''
    |- inputs:
        |-- date: date from which the degree for all the nodes is going to be extracted.
        |-- community: data frame with the community daily PAL.

    |- outputs
        |-- dictCounter: dictionary where keys are the ids of the nodes and values are
                         the degree of the node in the specified date.
                         Example: {409:9, 123:3, ...}
    '''
    dictCounter = {}
    date = pd.to_datetime(date)
    # Initialize all the nodes with 0
    for node in nodes:
        dictCounter[node] = 0
        
    for edge, edge_date in edges:
        edgeSource = int(edge[0])
        edgeTarget = int(edge[1])
        '''Dropout is related to the first plan.
        Some nodes have data even after they have
        dropped out of the experiment.
        I kept the network as it is regarding the
        connections made in the system.

        if nodes[edge[0]]['dropout'] is not None:
            if nodes[edge[0]]['dropout'] <= date:
                print 'node dropped out', edge[0], edge_date
                continue
        if nodes[edge[1]]['dropout'] is not None:
            if nodes[edge[1]]['dropout'] <= date:
                continue
        '''

        if edge_date <= date:
            # Check dropout dates
            # if nodes[edge[0]]['dropout'] is None and nodes[edge[1]]['dropout'] is None:                
            if edgeSource in dictCounter:
                dictCounter[edgeSource] += 1
            else:
                dictCounter[edgeSource] = 1

            if edgeTarget in dictCounter:
                dictCounter[edgeTarget] += 1
            else:
                dictCounter[edgeTarget] = 1

    return dictCounter

def intersect(a, b):
    return list(set(a) & set(b))


def genCleandata(community, noncommunity, graph):
    '''
    |- inputs:
        |-- community: data frame with the daily PAL data for community participants.
        |-- noncommunity: data frame with the daily PAL data for non community participants.
        |-- graph: dictionary containing the information of the graph (nodes and edges).
    |- outputs:
        |-- community: data frame with the daily PAL and the attributes of all nodes remaining after cleaning.
        |-- noncommunity: data frame with the daily PAL and the attributes of all nodes remaining after cleaning.
        |-- nodesDegrees: data frame with the dates and the degrees for all the remaining nodes in the graph.
        |-- nodes: dictionary with the nodes information. Keys are the id and the values are the attributes.
        |-- new_edges: list of tuples containing ((source, target), date).
    '''

    print('Starting cleaning of the data...')
    print('---------------------------------------------')
    listOfNodes = graph['nodes']['node']
    
    print('\nOriginal number of NODES in gephi file: #', len(listOfNodes))
    print('---------------------------------------------')
    print('Creating a dictionary for the nodes...')
    nodesWithoutAtt = []
    nodes = {}
    for node in listOfNodes:
        # first_plan_start_date
        first_plan_start_date = None
        dropout = None
        gender = None
        corporation = None
        country = None
        bmi = None
        try:
            attributes = node['attvalues']['attvalue']
        except:
            nodesWithoutAtt.append(node['@id'])
            continue

        for att in attributes:        
            if att['@for'] == '1':
                if att['@value'] == '':
                    first_plan_start_date = None
                else:
                    first_plan_start_date = pd.to_datetime(att['@value'])

            if att['@for'] == '2':
                if att['@value'] != '':
                    dropout = pd.to_datetime(att['@value'])
                else: 
                    dropout = None
            if att['@for'] == '3':
                gender = att['@value']
                
            if att['@for'] == '4':
                corporation = att['@value']
            
            if att['@for'] == '5':
                country = att['@value']
            
            if att['@for'] == '6':
                bmi = att['@value']
            
        nodes[node['@id']] = {'first_plan': first_plan_start_date,
                                'dropout': dropout,
                                'gender': gender,
                                'corporation': corporation,
                                'country': country,
                                'bmi': bmi
                               }
    print('Nodes without attributes removed from the graph: \t#', len(nodesWithoutAtt))
    print('Remaining nodes in the graph: \t\t\t#', len(nodes.keys()))

    print('---------------------------------------------')
    print('Creating a list for the edges...')

    listOfEdges = graph['edges']['edge']
    edges = []
    
    for edge in listOfEdges:
        edgeSource = edge['@source']
        edgeTarget = edge['@target']
        
        edgeDate = edge['attvalues']['attvalue']['@value']
        if edgeDate == '':
            edgeDate = None
        else:
            edgeDate = pd.to_datetime(edgeDate)
        
        edges.append(((edgeSource, edgeTarget), edgeDate))
    print('Original number of EDGES in the original gephi file: #', len(edges))
    print('---------------------------------------------')

    print('Remove nodes with no attributes from graph\n\
           Nodes to be removed:\n\
           1. No attributes at all\n\
           2. No corporation\n\
           3. No gender\n\
           4. No first plan date\n\
           5. Dropout date before the beggining of the experiment\n\
           6. Start date after end of the time range\n\
           Special inclusion: *the plan takes 84 days*.')

    # initial and end dates with useful data for the analysis
    init = pd.Timestamp('2010-04-28')
    end = pd.Timestamp('2010-08-01')

    nodes2clean = []
    miss_data = []
    fp_out = []
    drop = []

    for node in nodes:
        # Missing data
        if nodes[node]['bmi'] is None or nodes[node]['corporation'] is None or \
            nodes[node]['first_plan'] is None or nodes[node]['gender'] is None:
            nodes2clean.append(node)
            miss_data.append(node)
        elif nodes[node]['first_plan'] > end:
            nodes2clean.append(node)
            fp_out.append(node)
        elif nodes[node]['dropout'] != None and nodes[node]['dropout']< init:
            nodes2clean.append(node)
            drop.append(node)

    print('TOTAL NODES TO CLEAN: \t\t\t#', len(nodes2clean))
    print('Nodes missing data: \t\t\t#',len(miss_data))
    print('Nodes with first plan date > end date: \t#', str(len(fp_out)))
    print('Nodes with drop out date < init date: \t#', str(len(drop)))

    for node in nodes2clean:
        del nodes[node]

    print('Remaining nodes with attributes: #', len(nodes.keys()))
    print('---------------------------------------------')

    print('Remove nodes not present in the intersection between the gephi and PAL files...')
    # community nodes in csv
    com_nodes_csv = list(set(community.id))
    com_nodes_graph = list(map(int, nodes.keys()))

    print('Nodes in the daily PAL file: \t\t#', len(com_nodes_csv))
    print('Nodes in the gephi file: \t\t#', len(com_nodes_graph))

    nodes_in_both = intersect(com_nodes_csv, com_nodes_graph)

    print('Nodes in both files: \t\t\t#', len(nodes_in_both))
    print('.......................................................')

    nodes2remove_in_graph = list(set(com_nodes_graph) - set(nodes_in_both))

    print('Nodes to remove in the graph (Nodes present in\n\
        the graph and not in the csv file): \t#', len(nodes2remove_in_graph))

    nodes2remove_in_csv = list(set(com_nodes_csv) - set(nodes_in_both))

    print('Nodes to remove in the daily PAL archive (Nodes present in\n\
        the csv file and not in the graph file): \t#', len(nodes2remove_in_csv))
    print('.......................................................')

    # Remove in dictionary
    for node in nodes2remove_in_graph:
        del nodes[str(node)]

    # Remove in the DataFrame
    community = community[~community['id'].isin(nodes2remove_in_csv)]

    print('Remaining nodes in the gephi file: \t\t#', len(nodes.keys()))
    print('Remaining nodes in the csv (PAL) file: \t\t#', len(list(set(community.id))))
    print('---------------------------------------------')

    print('Cleaning edges... (Use anytime needed.)\n\
        Edges will be removed in case of:\n\
        \t1. one of the nodes is not present anymore\n\
        \t2. There is no start date for the edge\n\
        \t3. The reciprocal edge is not present\n\
        \t4. Edges started after end date')

    print('\nNumber of original edges in the gephi file: \t\t#', len(edges))
    print('Remaining nodes in the gephi file: \t\t#', len(nodes_in_both))

    new_edges = []
    # For removing the reciprocal ones
    edges_included = []
    edges_removed = []
    edges_no_date = []
    edges_no_node = []
    edges_reciprocal = []
    edges_after_end = []
    edge_rec_none = []

    for edge, start_date in edges:
        # No start date
        if start_date is None:
            edges_no_date.append(edge)
            edges_removed.append(edge)
        # If there is a None edge in the system (non reciprocal connection)
        elif ((edge[1], edge[0]), None) in edges:
            edge_rec_none.append(edge)
        # If one of the nodes does not exist anymore
        elif int(edge[0]) not in nodes_in_both or int(edge[1]) not in nodes_in_both:
            edges_no_node.append(edge)
            edges_removed.append(edge)
        # To garantee that the edges have same start date when reciprocal
        elif ((edge[1], edge[0]), start_date) in edges_included:
            edges_reciprocal.append((edge,start_date))
        # start date after the end of the experiment
        elif start_date > end:
            edges_after_end.append(edge)
        else:
            new_edges.append((edge,start_date))
            edges_included.append((edge, start_date))

    print('New edges (edges included): \t\t\t#', len(new_edges))
    print('Reciprocal edges: \t\t#', len(edges_reciprocal))
    print('Edges without a reciprocal peer: #', len(edge_rec_none))
    print('Edges witouth start date: \t\t#', len(edges_no_date))
    print('Edges with removed nodes: \t\t#', len(edges_no_node))
    print('Edges connected after end of time range: \t\t#', len(edges_after_end))

    setNodes = set()
    for edge, date in new_edges:
        setNodes.add(edge[0])
        setNodes.add(edge[1])
    print('Nodes present in the edges\' set: \t\t#', len(list(setNodes)))

    setNodes = map(int, list(setNodes))
    # All the nodes present in the edges are in the intersection with the nodes in both csv and graph
    print('Intersection of nodes in the gephi and csv \n\
        with nodes in the set of edges: \t\t#', len(intersect(list(setNodes), nodes_in_both)))
    print('---------------------------------------------')

    print('Creating one data frame with the information of the gephi + csv files...')

    # Create a DataFrame for the nodes characteristics
    nodes_df = pd.DataFrame(nodes).T
    nodes_df['id'] = pd.to_numeric(nodes_df.index)

    community = pd.merge(community, nodes_df, on='id', how='left')
    community = community[['id', 'date', 'pal', 'target_pal', 'minutes_moderate', 
                           'minutes_high', 'status', 'bmi', 'corporation', 'country', 
                           'dropout', 'gender', 'first_plan']]
    print('Head of the community data frame:')
    print(community.head())
    print('---------------------------------------------')

    print('Generating the degrees\' dictionary...')

    init = pd.to_datetime('2010-04-28')
    end = pd.to_datetime('2010-08-01')
    datelist = pd.date_range(init, periods=96).tolist()

    dailyDegreeDict = {}

    for day in datelist:
        dailyDegreeDict[day] = getNodesDegreeAtDate(day, list(set(community.id)), new_edges)

    nodesDegrees = pd.DataFrame(dailyDegreeDict).T
    print('Nodes\' degrees data frame head:')
    print(nodesDegrees.head())
    print('---------------------------------------------')

    print('Data from non community data set processing...')
    print('(Should become a function in the future)')

    # Non community people start plan date
    subnoncommunity = noncommunity[['id', 'status', 'date']].copy()
    subnoncommunity.date = pd.to_datetime(subnoncommunity.date)

    # This part of the code will generate the dictionary with the
    # estimated first plan date for the nodes.

    start_plan_date_dict = {}

    for index, row in subnoncommunity.iterrows():
        #print row['c1'], row['c2']
        if row['status'] == 'in_plan':
            # Get the old value stored or None in case it is 
            # the first time this id is read
            dt = start_plan_date_dict.get(row['id'], None)

            if dt == None:
                start_plan_date_dict[row['id']] = row['date']
            else:
                if dt > row['date']:
                    start_plan_date_dict[row['id']] = row['date']

    start_plan_date_df = pd.DataFrame.from_dict(start_plan_date_dict, 'index')
    start_plan_date_df['id'] = start_plan_date_df.index
    start_plan_date_df.reset_index(inplace=True, drop=True)
    start_plan_date_df.columns = ['first_plan', 'id']
    start_plan_date_df = start_plan_date_df[['id', 'first_plan']]

    noncommunity = pd.merge(noncommunity, start_plan_date_df, on='id', how='left')
    noncommunity = noncommunity[['id', 'corporation', 'gender', 'date', 'pal',
           'target_pal', 'minutes_moderate', 'minutes_high', 'status',
           'first_plan']]

    noncommunity.gender = noncommunity.gender.replace('female', 'F')
    noncommunity.gender = noncommunity.gender.replace('male', 'M')
    print('Non community DF head:')
    print(noncommunity.head())

    return community, noncommunity, nodesDegrees, nodes, new_edges


def readCleandata(data_f):
    '''
    Test if the clean data is available and return it.
    '''
    clean_files_f = data_f+'clean/'
    if not os.path.exists(clean_files_f):
        print('Clean data folder ({}clean) does not exist. Run cleandata() to generate it.'.format(data_f))
        return False
    try:
        community_f = clean_files_f+'community.csv'
        noncommunity_f = clean_files_f+'noncommunity.csv'
    except:
        print('Files community.csv and/or noncommunity.csv not generated. Run cleandata() to generate it.')
        return False

    # Community data set
    print('-----------------------------------------------------')
    print('community.csv and noncommunity.csv read successfully!')
    print('-----------------------------------------------------')
    community = pd.read_csv(community_f)
    community.date = pd.to_datetime(community.date)
    community.first_plan = pd.to_datetime(community.first_plan)
    community['end_first_plan'] = community['first_plan'] + pd.DateOffset(days=84)
    community.drop('Unnamed: 0', axis=1, inplace=True)

    noncommunity = pd.read_csv(noncommunity_f)
    noncommunity.date = pd.to_datetime(noncommunity.date)
    noncommunity.first_plan = pd.to_datetime(noncommunity.first_plan)
    noncommunity['end_first_plan'] = noncommunity['first_plan'] + pd.DateOffset(days=84)
    noncommunity.drop('Unnamed: 0', axis=1, inplace=True)

    # Read nodesDegrees
    try:
        nodesDegrees = pd.read_csv(clean_files_f+'degrees_per_day.csv')
    except:
        print('File degrees_per_day.csv not generated. Run cleandata() to generate it.')
        return False
    nodesDegrees.index = pd.to_datetime(nodesDegrees['Unnamed: 0'])
    nodesDegrees.drop('Unnamed: 0', axis=1, inplace=True)
    nodesDegrees.index.names = [None]

    print('-----------------------------------------------------')
    print('degrees_per_day.csv read successfully!')
    print('-----------------------------------------------------')

    try:
        nodes = pickle.load(open(clean_files_f+'nodes.pickle', 'rb'))
        edges = pickle.load(open(clean_files_f+'edges.pickle', 'rb'))
    except:
        print('Files nodes.pickle and/or edges.pickle not generated. Run cleandata() to generate it.')
        return False

    print('-----------------------------------------------------')
    print('nodes.pickle and edges.pickle read successfully!')
    print('-----------------------------------------------------')

    nodes_in_community = list(set(community.id))
    print('\tNODES IN COMMUNITY DATA SET: \t\t', len(nodes_in_community))

    nodes_in_noncommunity = list(set(noncommunity.id))
    print('\tNODES IN NON COMMUNITY DATA SET: \t', len(nodes_in_noncommunity))

    return community, noncommunity, nodesDegrees, nodes, edges




def getCleandata(data_f='../data/'):

    outputs = readCleandata(data_f)

    if outputs is not False:
        community, noncommunity, nodesDegrees, nodes, edges = outputs
    else:
        print('Running cleandata() to generate clean data set.')
        raw_files_f=data_f+'raw/'
        destine_f=data_f+'clean/'

        if not os.path.exists(destine_f):
            os.makedirs(destine_f)

        community_f = raw_files_f+'activity_per_day.csv'
        noncommunity_f = raw_files_f+'noncommunity_activity_per_day.csv'

        community_df = pd.read_csv(community_f)
        noncommunity_df = pd.read_csv(noncommunity_f)

        # non community does not have duplicates
        community_df = community_df.drop_duplicates()

        # Graph
        graph_file = raw_files_f+'communalitics2015.gexf'

        with open(graph_file) as fd:
            root = xmltodict.parse(fd.read())
        graph = root['gexf']['graph']

        community, noncommunity, nodesDegrees, nodes, edges = genCleandata(community_df, noncommunity_df, graph)

        # Adjustments 
        community.date = pd.to_datetime(community.date)
        community.first_plan = pd.to_datetime(community.first_plan)
        community['end_first_plan'] = community['first_plan'] + pd.DateOffset(days=84)
        noncommunity.date = pd.to_datetime(noncommunity.date)
        noncommunity.first_plan = pd.to_datetime(noncommunity.first_plan)
        noncommunity['end_first_plan'] = noncommunity['first_plan'] + pd.DateOffset(days=84)

        # Saving files...
        print('Saving files...')
        community.to_csv(destine_f+'community.csv')
        noncommunity.to_csv(destine_f+'noncommunity.csv')
        nodesDegrees.to_csv(destine_f+'degrees_per_day.csv')
        pickle.dump(nodes , 
            open(destine_f+'nodes.pickle', 'wb'))

        pickle.dump(edges, 
            open(destine_f+'edges.pickle', 'wb'))
    
    print('----------------------------------------------')
    print('Final Report')
    print('----------------------------------------------')

    print('Number of nodes in community csv file: #', len(set(community.id)))
    print('Number of nodes in non community csv file: #', len(set(noncommunity.id)))
    print('Number of nodes in nodes list: #', len(set(nodes)))
    print('Number of edges in edges list: #', len(set(edges)))

    return community, noncommunity, nodesDegrees, nodes, edges


if __name__ == '__main__':
    getCleandata()