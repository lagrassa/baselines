def get_formatted_name(params):
    name =  params['alg']+params['env_name'] + params['exp_name']+"obs_"+str(params['obs_noise_std'])+"act_"+str(params['action_noise_std'])
    if "goal_radius" in params.keys():
        name += "rw_"+str(params["goal_radius"])
    if "rew_noise_std" in params.keys():
        name += "rew_noise_std_"+str(params["rew_noise_std"])

    return name 

def get_short_form_name(params):
    name =  params['alg']+params['env_name'] + params['exp_name']
    return name

