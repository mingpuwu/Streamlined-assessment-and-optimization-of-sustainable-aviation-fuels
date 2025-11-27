import joblib
import pandas as pd

def use_model(params, models, zscores, zero_std_index, personIndexs):
    reshape_params = params.reshape(1,-1)

    index_zero_std_params = reshape_params[:,zero_std_index]
    predicted_ys = []
    for model,personindex,zscore in zip(models,personIndexs,zscores):

        person_x = index_zero_std_params[:,personindex]

        zscoreNewx = zscore.transform(person_x)
        zscoreNewx = zscoreNewx.astype(float)

        predicted_y = model.predict(zscoreNewx)
        predicted_y = predicted_y.flatten()[0]
        predicted_ys.append(predicted_y)
    
    return predicted_ys

if __name__ == '__main__':
    allY = ['density_mean','derived_cetane_number_mean',
        'Distillation0_mean','Distillation10_mean',
        'Distillation20_mean','Distillation50_mean',
        'Distillation90_mean','Distillation100_mean',
        'flash_point_mean','freezing_point_mean',
        'net_heat_of_combustion_mean','surface_tension_mean',
        'viscosity_kinematic-20_mean','viscosity_kinematic-40_mean']
    
    indexdf = pd.read_excel('models/zero_std_index.xlsx')
    indexdfValues = indexdf['zero_std_index'].values

    df = pd.DataFrame()
    readdata = pd.read_excel('sumdata.xlsx')
    tetssum = [0]
    for diff_x_index in tetssum:
        diff_x = 'sumdata_'+str(diff_x_index)
        params = readdata[diff_x].values
        params = params.reshape(1,-1)

        models = []
        person_index_s = []
        zscores = []
        for oneYname in allY:
            model = joblib.load('models/'+oneYname+'_stacking_model.pkl')
            models.append(model)

        for oneYname in allY:
            zscore = joblib.load('models/'+oneYname+'_zscore_mode.pkl')
            zscores.append(zscore)

        for oneYname in allY:
            read_person_index = pd.read_excel('models/'+oneYname+'_person_index.xlsx')['zero_std_index'].values
            read_person_index = read_person_index.flatten()
            person_index_s.append(read_person_index)

        PreYs = use_model(params, models, zscores, indexdfValues, person_index_s)

        print(PreYs)
