from lib_interface import (np, pd, sns, showDf, plt, 
                            rcParams, tsa, decomposition, 
                            warnings, style, datetime, file_manip)

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'


style.use('ggplot')

#### fontions (début) ####

def csv_info(*args, **kwargs):
    df = kwargs.get('df', None)
    col = kwargs.get('col', None)
    info = kwargs.get('info', None)
    if df is None:df="train.csv"  
    if col is None:col="Date"  
    csv = pd.read_csv(file_manip.inputPath()+df, skipinitialspace=True, usecols=[col])
    if info == "len":return len(csv.index)
    if info =="date":return max(csv['Date'])
    if info is None: return max(csv[col])


def filter(vdf, m, d):
    #Trie les données celon le departement et le magasin
    if d==0 :
        df = vdf.query("Store=='"+str(m)+"'").reset_index(drop=True).sort_values(by='Date')
    else :
        df = vdf.query("Store=='"+str(m)+"' and Dept=='"+str(d)+"'").reset_index(drop=True).sort_values(by='Date')
    return df.drop(columns=['Store', 'Dept']) #enlève les données redondants

def showgraph(vdf, xaxis, yaxis, xlabel, ylabel, title):
    vdf = vdf.set_index(xaxis)
    
    for col in np.nditer(yaxis):
        vdf[str(col)].plot()
    plt.legend(loc=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def show_decompose(df, *args, **kwargs):
    focused = kwargs.get('focused', None)
    model = kwargs.get('model', None)
    if focused is None:focused="Weekly_Sales"
    y = df[['Date', focused]].copy()
    y = y.set_index('Date')
    rcParams['figure.figsize'] = 18, 8 
    decomposition = tsa.seasonal_decompose(y, model=model)
    fig = decomposition.plot()
    plt.gcf().canvas.set_window_title(model.upper())
    plt.show()

def show_dept_sale(bdd, store, dept_min, dept_max):
    vdf = filter(bdd,store,0)[["Date"]].drop_duplicates().sort_values(by="Date")
    cols = np.array([], dtype = "S5")
    for x in range(dept_min,dept_max):
        deptsales = 'S'+str(store)+'_D'+str(x)
        ### Fonction filter(dataframe, storeNumber, deptNumber) ###
        vdf = vdf.merge(filter(bdd,store,x)[["Date", "Weekly_Sales"]].rename(columns={"Weekly_Sales": deptsales}))
        cols = np.append(cols, [deptsales])
#  showgraph(vdf,'Date',cols, 'Date', 'Revenue', 'Store sales')

def prep_features(bdd):
    for i, row in bdd.iterrows():
        bdd.at[i,'Fuel_Price'] = round(row['Fuel_Price'], 2)
        bdd.at[i,'Temperature'] = round(((row['Temperature']-32) * 5/9), 2)
    bdd.fillna(0, inplace=True)


def manip_data(store, dept, *args, **kwargs):
    train_set = kwargs.get('train_set', None)
    test_set = kwargs.get('test_set', None)
    ftrs_set = kwargs.get('features_set', None)
    if train_set is None:train_set="train.csv"
    if test_set is None:test_set="test.csv"
    if ftrs_set is None:ftrs_set="features.csv"
    
    inpath = file_manip.inputPath()
    v_p_train = pd.read_csv(inpath+train_set) ### les ventes précèdentes de la compagnie ###
    v_p_test = pd.read_csv(inpath+test_set) ### les données qui doivent être générer ###
    facteurs = pd.read_csv(inpath+ftrs_set) ### les facteurs influants sur la vente ###
    
    train = v_p_train.merge(facteurs, how='left')
    test = v_p_test.merge(facteurs, how='left')

    train['Date'] =pd.to_datetime(train.Date)
    test['Date'] =pd.to_datetime(test.Date)
    train['Month'] = pd.to_datetime(facteurs['Date']).dt.month
    test['Month'] = pd.to_datetime(facteurs['Date']).dt.month
    SD_train = filter(train,store,dept)
    SD_test = filter(test,store,dept)

    prep_features(SD_train)
    prep_features(SD_test)
    return SD_train, SD_test


def corr_heatmap(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Heat Map')
    plt.show()

def returnFeatures():
    facteurs = pd.read_csv(file_manip.inputPath()+"features.csv").drop(columns=['Store'])
    facteurs['Date'] =pd.to_datetime(facteurs.Date)
    facteurs['Month'] = pd.to_datetime(facteurs['Date']).dt.month
    facteurs = facteurs.drop(columns=['Date'])
    return sorted(facteurs)
#### fontions (fin) ####