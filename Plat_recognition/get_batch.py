# 分批
def get_batch(batch_size,input_count):
    batch_size=batch_size
    batch_count = int(input_count / batch_size)
    remainder = input_count % batch_size
    print('The training data set is divided into %s batches, '
          'the previous batch of %s data, and the last batch of %s data' % (batch_count,batch_size,remainder))
    return batch_count,remainder













