import deep_visual_geo_localization_benchmark as gl

# train should take all arguments necessary for training and return trained network
# this is where custom methods to train networks can be used

# training for me is going to only be from the geo-localization benchmark
# this means some stuff will probs be redundant if own training is implemented


def train(args):

    model = gl.train.train(args)

    return model

