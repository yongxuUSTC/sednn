
import numpy as np
#from supports import to_list


# truncate seq or pad with 0, input can be list or np.ndarray
# the element in x should be ndarray, then pad or trunc all elements in x to max_len
# type: 'post' | 'pre'
# return x_new (N*ndarray), mask(N*max_len)
def pad_trunc_seqs( x, max_len, pad_type='post' ):
    list_x_new, list_mask = [], []
    for e in x:
        L = len( e )
        e_new, mask = pad_trunc_seq( e, max_len, pad_type )
        list_x_new.append( e_new )
        list_mask.append( mask )
    
    type_x = type( x )
    if type_x==list:
        return list_x_new, list_mask
    elif type_x==np.ndarray:
        return np.array( list_x_new ), np.array( list_mask )
    else:
        raise Exception( "Input should be list or ndarray!" )

   
# pad or trunc seq, x should be ndarray
# return x_new (ndarray), mask (1d array)
def pad_trunc_seq( x, max_len, pad_type='post' ):
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len-L,) + shape[1:]
        pad = np.zeros( pad_shape )
        if pad_type=='pre': 
            x_new = np.concatenate( (pad, x), axis=0 )
            mask = np.concatenate( [ np.zeros(max_len-L), np.ones(L) ] )
        elif pad_type=='post': 
            x_new = np.concatenate( (x, pad), axis=0 )
            mask = np.concatenate( [ np.ones(L), np.zeros(max_len-L) ] )
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    else:
        if pad_type=='pre':
            x_new = x[L-max_len:]
            mask = np.ones( max_len )
        elif pad_type=='post':
            x_new = x[0:max_len]
            mask = np.ones( max_len )
        else:
            raise Exception("pad_type should be 'post' | 'pre'!")
    return x_new, mask

# enframe sequence to matrix
def enframe( x, win, inc ):
    Xlist = []
    p = 0
    while ( p+win <= len(x) ):
        Xlist.append( x[p:p+win] )
        p += inc
    
    X = np.array( Xlist )
    return X

# concatenate feautres     
def mat_2d_to_3d( X, agg_num, hop ):
    # pad to at least one block
    len_X, n_in = X.shape
    if ( len_X < agg_num ):
        X = np.concatenate( ( X, np.zeros((agg_num-len_X, n_in)) ) )
        
    # agg 2d to 3d
    len_X = len( X )
    i1 = 0
    X3d = []
    while ( i1+agg_num <= len_X ):
        X3d.append( X[i1:i1+agg_num] )
        i1 += hop
    return np.array( X3d )


# concatenate feautres     
def mat_2d_to_3d_paddingzeros( X, agg_num, hop ):
    # pad to at least one block
    len_X, n_in = X.shape
    if ( len_X < agg_num ):
        X = np.concatenate( ( X, np.zeros((agg_num-len_X, n_in)) ) )
        
    # agg 2d to 3d
    len_X = len( X )
    i1 = 0
    X3d = []


    ### replaced by zero padding when reading X
    ##padding zeros in the left begining with (agg_num-1)/2 zeros:
    #X = np.concatenate( ( np.zeros(((agg_num-1)/2, n_in)), X  ) )
    ##padding zeros in the right end with (hop-1)/2 zeros:
    #X = np.concatenate( ( X, np.zeros(((agg_num-1)/2, n_in))  ) )
    #len_X=len(X) #update len of X after padding zeros

    while ( i1+agg_num <= len_X ):
        X3d.append( X[i1:i1+agg_num] )
        i1 += hop
    return np.array( X3d )

# convert from 3d to 4d, for input of cnn        
def reshape_3d_to_4d( X ):
    [ N, n_row, n_col ] = X.shape
    return X.reshape( (N, 1, n_row, n_col) )
    
    
# sparse label to categorical label
# x is 1-dim ndarray
def sparse_to_categorical( x, n_out ):
    x = x.astype(int)   # force type to int
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros( (N,n_out) )
    x_categ[ np.arange(N), x ] = 1
    return x_categ.reshape( (shape)+(n_out,) )
