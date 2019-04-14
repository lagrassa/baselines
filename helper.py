def get_formatted_name(params):
    name =  params['alg']+params['env_name'] + params['exp_name']+"obs_"+str(params['obs_noise_std'])+"act_"+str(params['action_noise_std'])
    if "goal_radius" in params.keys():
        name += "rw_"+str(params["goal_radius"])
    return name 
