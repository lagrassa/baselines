def get_formatted_name(params):
    return params['alg']+params['env_name'] + params['exp_name']+"obs_"+str(params['obs_noise_std'])+"act_"+str(params['action_noise_std'])