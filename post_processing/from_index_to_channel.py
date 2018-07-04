def from_index_to_channel(list_of_channels):
    pkl_file = open('index_to_channel.pkl', 'rb')
    mydict = pk.load(pkl_file)
    channel_names=[]
    for key in list_of_channels:
        channel_names.append(mydict[key])
    return channel_names