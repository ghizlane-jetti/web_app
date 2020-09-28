# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import optimize
from datetime import timedelta
from scipy import stats as sps
from PIL import Image
from scipy.signal import argrelextrema
import math as m

#pip install openpyxl

#Title
st.title("🦠 Covid-19 Situation")



#Covid data
data_cov=pd.read_excel("C:/Users/hp/Desktop/Covid.xlsx")
data_cov1=data_cov.copy()
#Country/Region
c=list((data_cov["Country/Region"]))
c=list(np.unique(c))

#select0= st.sidebar.selectbox('Select :', ["Overview","By country"], key='2')
#if select0=="By country":

#Select country:

select1= st.sidebar.selectbox('Select Country/Region:', c, key='2')

s=select1
st.write("#### :globe_with_meridians: Country selected: **{}**".format(s))

#Dates
data_cov=data_cov[data_cov['Country/Region']==select1]
dates = list(data_cov["Date"])
d=[]
for i in dates:

    d.append(datetime.strptime(i, '%d/%m/%Y'))

#Select Confirmed/Recovered/Deaths
select3  = st.selectbox('Select :', ['Confirmed', 'Recovered',"Deaths","Difference","Closure","Actif"])


#Confirmed, Recovered and death list
conf=list(data_cov['Confirmed'])

rec=list(data_cov['Recovered'])

death=list(data_cov['Deaths'])

#Country and Province list:
coun=[select1 for i in range(len(d))]


#Difference cases:
Diff=list(data_cov['Difference'])

#Closed cases:
clo=list(data_cov['Closure'])

#Actif cases:
act=list(data_cov['Actif'])


#Dataframe:

df = pd.DataFrame(list(zip(d,coun,conf,rec,death,Diff,clo,act)),
            columns =['Date',"Country/Region","Confirmed","Recovered","Deaths","Difference","Closure","Actif"])
if select3 == 'Confirmed':
    fig = px.bar(df,x="Date", y="Confirmed")
    fig.update_layout(

        xaxis_title="Time (Date)",
        yaxis_title="Nº of Confirmed cases")
    st.plotly_chart(fig)

elif select3 == "Recovered":
    fig = px.bar(df,x="Date", y="Recovered")
    fig.update_layout(

        xaxis_title="Time (Date)", yaxis_title="Nº of Recovered cases")
    st.plotly_chart(fig)
elif select3 == "Deaths":
    fig = px.bar(df,x="Date", y="Deaths")
    fig.update_layout(

        xaxis_title="Time (Date)",
        yaxis_title="Nº of Deaths",

    )

    st.plotly_chart(fig)
elif select3 == "Actif":
    fig = px.bar(df,x="Date", y="Actif")
    fig.update_layout(

        xaxis_title="Time (Date)",
        yaxis_title="Nº of actif cases",

    )

    st.plotly_chart(fig)
elif select3 == "Closure":
    fig = px.bar(df,x="Date", y="Closure")
    fig.update_layout(

        xaxis_title="Time (Date)",
        yaxis_title="Nº of Closure cases",

    )

    st.plotly_chart(fig)
elif select3 == "Difference":
    fig = px.bar(df,x="Date", y="Difference")
    fig.update_layout(

        xaxis_title="Time (Date)",
        yaxis_title="Nº of new cases per day",

    )

    st.plotly_chart(fig)

st.markdown('**Cumulative Confirmed cases:** {}  \n'.format(list(df['Confirmed'])[-1]) +
            '**Cumulative deaths:** {}  \n'.format(list(df['Deaths'])[-1])+
            '**Cumulative Recovered cases:** {}'.format(list(df['Recovered'])[-1]) )

# Load  data
data = df

st.subheader('Data')
st.write(data)

#Downoad data

df.to_excel("data.xlsx")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)


#Lethality rate
st.subheader("☞ Evolution of Lethality over time in "+select1)

try:
    n=len(clo)
    letal1=[]
    letal2=[]
    for i in range(30,n):
        letal1.append(100*(death[i]/(clo[i])))
        letal2.append(100*(death[i]/(conf[i])))
    l1=[]
    l2=[]
    for i in range(30):
        l1.append(np.nan)
    l1.extend(letal1)
    letal1=l1
    for i in range(30):
        l2.append(np.nan)
    l2.extend(letal2)
    letal2=l2
    letalite=pd.DataFrame(list(zip(list(df['Date']),list(df['Country/Region']),letal1,letal2)),columns
    =['Date',"Country/Region",'Deaths/Closure','Deaths/Confirmed'])
    fig = px.line(letalite,x="Date", y=['Deaths/Closure','Deaths/Confirmed'])

    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="Lethality"
        )

    st.plotly_chart(fig)
    #Downoad data

    letalite.to_excel("lethality.xlsx")

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html('lethality.xlsx', 'Lethality_Data'), unsafe_allow_html=True)
except:
    print("Something went wrong")

#Recovery rate:

st.subheader("☞ Evolution of Recovery rate over time in "+select1)
try:
    letal=[]
    for i in range(30,n):
        letal.append(100*(rec[i]/(clo[i])))
    l=[]
    for i in range(30):
        l.append(np.nan)
    l.extend(letal)
    letal=l
    recovery=pd.DataFrame(list(zip(list(df['Date']),list(df['Country/Region']),letal1)),columns =['Date',"Country/Region",'Deaths/Closure'])
    recovery["Recovery"]=letal
    fig = px.line(recovery,x="Date", y=['Recovery','Deaths/Closure'])
    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="Recovery rate"
        )

    st.plotly_chart(fig)
    #Downoad data

    recovery.to_excel("recovery.xlsx")

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html('recovery.xlsx', 'Recovery_data'), unsafe_allow_html=True)
except:
    print("Something went wrong")


# Logistic Model

try:
    df.index=[i for i in range(len(df))]

    x = df["Confirmed"].index
    y =df["Confirmed"].values

    # inflection point estimation
    dy = np.diff(y,n=10) #  derivative
    idx_max_dy = np.argmax(dy)

    #Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) )

    def logistic_f1(X, c, k, m):
        y = c / (1 + np.exp(-k*(X-m)))
        return y
    # optimize from scipy


    logistic_model1, cov = optimize.curve_fit(logistic_f1,
                                    xdata=np.arange(len(df["Confirmed"])+np.argmax(dy)-len(df)),
                                    ydata=df["Confirmed"].values[0: np.argmax(dy)],
                                    maxfev=10000,
                                    p0=[np.max(list(df["Confirmed"])[0: np.argmax(dy)]), 1, 1])


    #Logistic function: f(x) = a / (1 + e^(-b*(x-c)))

    def f(x):
        return logistic_model1[0] / (1 + np.exp(-logistic_model1[1]*(x-logistic_model1[2])))

    y_logistic1 = f(x=np.arange( np.argmax(dy)))

    #Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) )

    def logistic_f2(X, c, k, m):
        y = c / (1 + np.exp(-k*(X-m)))
        return y
    # optimize from scipy


    logistic_model2, cov = optimize.curve_fit(logistic_f2,
                                    xdata=np.arange(len(df["Confirmed"])- np.argmax(dy)),
                                    ydata=df["Confirmed"].values[ np.argmax(dy):],
                                    maxfev=10000,
                                    p0=[np.max(list(df["Confirmed"])[ np.argmax(dy):]), 1, 1])

    #Logistic function: f(x) = a / (1 + e^(-b*(x-c)))

    def f(x):
        return logistic_model2[0] / (1 + np.exp(-logistic_model2[1]*(x-logistic_model2[2])))

    y_logistic2 = f(x=np.arange( len(df)-np.argmax(dy)+60)) # 60 ==> Prédiction des cas confirmés dans les futurs 2 mois.
    confirm=list(df["Confirmed"])
    no=[np.nan for i in range(len(df),len(df)+60)]
    confirm.extend(no)
    index=[i for i in range(len(df)+60)]
    log=[]
    log.extend(y_logistic1)
    log.extend(y_logistic2)
    date=d

    fin=date[-1]
    for i in range(60):
        k=fin+timedelta(days=i+1)
        date.append(k)
    log_df = pd.DataFrame(list(zip(index,date, confirm,log)),
                columns =['index', "Date",'Confirmed',"Predicted"])
    st.subheader("☞ Logistic Model "+select1)

    fig = px.scatter(log_df,x="Date", y='Confirmed')

    reference_line = go.Scatter(x=date,
                                y=log,
                                mode="lines",
                                line=go.scatter.Line(color="red"),
                                name="Predicted",
                                showlegend=True)
    fig.add_trace(reference_line)
    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="Logistic"
        )

    st.plotly_chart(fig)
except:
    print("Something went wrong")


#R0 simplist method
st.subheader("📈 Evolution of R0 using Simplist Method in "+select1)

try:
    data = pd.DataFrame(df, columns=['Date','Difference']).set_index('Date')
    cases=np.array(list(data['Difference']))
    indi=list(np.where(cases == 0)[0])
    diff_indi=np.diff(indi)
    ind=data.index
    data['Date']=list(df["Date"])
    l=[]
    j=0
    for i in diff_indi:
        j=j+1
        if i>15:
            l.append(indi[j-1])
            l.append(indi[j])
    if len(l)==0:

        data2=data[indi[-1]:]
        if len(data2)<15:
            st.write("#### Nothing to show")
        else:
            data2['smooth_mean(gauss, win=4)'] = data2.iloc[:,0].rolling(4,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()

            data2['smooth_mean(gauss, win=7)'] = data2.iloc[:,0].rolling(7,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()
            gauss=list(data2['smooth_mean(gauss, win=7)'])[9:]
            l_7=[]
            for i in range(len(gauss)-6):
                l_7.append(round(gauss[i+6])/round(gauss[i]))

            gauss=list(data2['smooth_mean(gauss, win=7)'])[9:]
            l_4=[]
            for i in range(len(gauss)-3):
                l_4.append(round(gauss[i+3])/round(gauss[i]))
            N=len(data2)
            n_4=len(l_4)
            n_7=len(l_7)

            gauss_4=[]
            gauss_7=[]
            for i in range(N-n_4):
                gauss_4.append(np.nan)
            gauss_4.extend(l_4)

            for i in range(N-n_7):
                gauss_7.append(np.nan)
            gauss_7.extend(l_7)
            data2["Gaussian_R0_4_Days"]=gauss_4
            data2["Gaussian_R0_7_Days"]=gauss_7
            col = data2.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
            data2['R0_Simpliste'] = col.mean(axis=1)
            R0_sim=list(data2['R0_Simpliste'])
            data2.index=data2.Date
            from_date  = st.selectbox('From :', list(data2.Date) )
            ind1=list(data2.Date).index(from_date)
            l2=list(data2.Date)[int(ind1)+1:]
            to_date= st.selectbox('To :',l2  )
            ind2=list(data2.Date).index(to_date)
            R0_sim=R0_sim[ind1:ind2+1]
            dt=list(data2.Date)[ind1:ind2+1]
            data_per=pd.DataFrame(list(zip(dt,R0_sim)),
                columns =['Date',"R0_Simpliste"])
            fig = px.line(data_per,x="Date", y="R0_Simpliste")

            fig.update_layout(

            xaxis_title="Date",
            yaxis_title="R0_Simp"
        )
            reference_line = go.Scatter(x=list(data_per['Date']),
                                y=[1 for i in range(len(dt))],
                                mode="lines",
                                line=go.scatter.Line(color="red"),

                                showlegend=False)
            fig.add_trace(reference_line)
            st.plotly_chart(fig)
            data2=data_per
            data2['Country']=[select1 for i in range(ind1,ind2+1)]
            da=[data2['Date'][i].strftime("%d-%m-%Y") for i in range(len(data2))]
            data2['Date']=da

            data2.plot(x="Date",y="R0_Simpliste",label="R0", figsize=(14,5), color="m")


            peak_indexes = argrelextrema(np.array(list(data2['R0_Simpliste'])), np.greater)
            peak_indexes = peak_indexes[0]
            plt.axhline(y=1,linestyle='--', color='black')

        # Plot peaks.
            peak_x = peak_indexes
            peak_y = np.array(list(data2['R0_Simpliste']))[peak_indexes]

        # Find valleys(min).
            valley_indexes = argrelextrema(np.array(list(data2['R0_Simpliste'])), np.less)
            valley_indexes = valley_indexes[0]


        # Plot valleys.
            valley_x = valley_indexes
            valley_y =  np.array(list(data2['R0_Simpliste']))[valley_indexes]

            reg_x=np.union1d(valley_indexes,peak_indexes)
            reg_y=np.array(list(data2['R0_Simpliste']))[reg_x]
            plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="Régression Linéaire")
            n=len(list(data2['R0_Simpliste']))
            data2['Country']=[select1 for i in range(n)]
            data2.to_excel("Data_R0_Simpliste_"+select1+".xlsx")
        # Save graph to file.
            plt.xlabel('Date')
            plt.legend(loc='best')

            plt.title("R0_Simpliste "+select1)
            plt.legend(loc='best')
            path=os.path.abspath(os.getcwd())
            plt.savefig(path+'\\R0_Sim.jpg')





            image = Image.open(path+'\\R0_Sim.jpg')
            st.image(image, caption='R0_Simplist '+select1,
            use_column_width=True)
        #Downoad data

        data_per.to_excel("R0_sim.xlsx")

        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
            return href
        st.markdown(get_binary_file_downloader_html('R0_sim.xlsx', 'R0_simp Data'), unsafe_allow_html=True)


    #if len(l)>1:
    else:
        n=len(l)
        #for i in range(n-1):
            #k1=l[i]
            #k2=l[i+1]
            #data1=data[k1:k2]
        k1=l[n-2]
        k2=l[n-1]
        data1=data[k1:k2]
        data1['smooth_mean(gauss, win=4)'] = data1.iloc[:,0].rolling(4,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()

        data1['smooth_mean(gauss, win=7)'] = data1.iloc[:,0].rolling(7,
            win_type='gaussian',
            min_periods=1,
            center=True).mean(std=2).round()
        gauss=list(data1['smooth_mean(gauss, win=7)'])[9:]
        l_7=[]
        for i in range(len(gauss)-6):
            l_7.append(round(gauss[i+6])/round(gauss[i]))

        gauss=list(data1['smooth_mean(gauss, win=7)'])[9:]
        l_4=[]
        for i in range(len(gauss)-3):
            l_4.append(round(gauss[i+3])/round(gauss[i]))
        N=len(data1)
        n_4=len(l_4)
        n_7=len(l_7)

        gauss_4=[]
        gauss_7=[]
        for i in range(N-n_4):
            gauss_4.append(np.nan)
        gauss_4.extend(l_4)

        for i in range(N-n_7):
            gauss_7.append(np.nan)
        gauss_7.extend(l_7)
        data1["Gaussian_R0_4_Days"]=gauss_4
        data1["Gaussian_R0_7_Days"]=gauss_7
        col = data1.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
        data1['R0_Simpliste'] = col.mean(axis=1)
        R0_sim=list(data1['R0_Simpliste'])
        data1.index=data1.Date
        from_date  = st.selectbox('From1 :', list(data1.Date) )
        ind1=list(data1.Date).index(from_date)
        l2=list(data1.Date)[int(ind1)+1:]
        to_date= st.selectbox('To :',l2  )
        ind2=list(data1.Date).index(to_date)
        R0_sim=R0_sim[ind1:ind2+1]
        dt=list(data1.Date)[ind1:ind2+1]
        data_per=pd.DataFrame(list(zip(dt,R0_sim)),
            columns =['Date',"R0_Simpliste"])

        fig = px.line(data_per,x="Date", y="R0_Simpliste")

        fig.update_layout(

            xaxis_title="Date",
            yaxis_title="R0"
        )
        reference_line = go.Scatter(x=list(data_per['Date']),
                                y=[1 for i in range(len(dt))],
                                mode="lines",
                                line=go.scatter.Line(color="red"),

                                showlegend=False)
        fig.add_trace(reference_line)
        data1=data_per
        data_per=[select1 for i in range(ind1,ind2+1)]
        da=[data1['Date'][i].strftime("%d-%m-%Y") for i in range(len(data1))]

        data1['Date']=da
        data1.plot(x="Date",y="R0_Simpliste",label="R0", figsize=(14,5), color="m")


        peak_indexes = argrelextrema(np.array(list(data1['R0_Simpliste'])), np.greater)
        peak_indexes = peak_indexes[0]
        plt.axhline(y=1,linestyle='--', color='black')

        # Plot peaks.
        peak_x = peak_indexes
        peak_y = np.array(list(data1['R0_Simpliste']))[peak_indexes]

        # Find valleys(min).
        valley_indexes = argrelextrema(np.array(list(data1['R0_Simpliste'])), np.less)
        valley_indexes = valley_indexes[0]


        # Plot valleys.
        valley_x = valley_indexes
        valley_y =  np.array(list(data1['R0_Simpliste']))[valley_indexes]

        reg_x=np.union1d(valley_indexes,peak_indexes)
        reg_y=np.array(list(data1['R0_Simpliste']))[reg_x]
        plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="linear regression")
        n=len(list(data1['R0_Simpliste']))
        data1['Country']=[select1 for i in range(n)]
        data1.to_excel("Data_R0_Simpliste_"+select1+".xlsx")
        # Save graph to file.
        plt.xlabel('Date')
        plt.legend(loc='best')

        plt.title("R0_Simpliste "+select1)
        plt.legend(loc='best')
        path=os.path.abspath(os.getcwd())
        plt.savefig(path+'\\R0_Sim.jpg')






        st.plotly_chart(fig)
        image = Image.open(path+'\\R0_Sim.jpg')
        st.image(image, caption='R0_Simplist '+select1,
            use_column_width=True)
        #Downoad data

        data_per.to_excel("R0_sim.xlsx")

        def get_binary_file_downloader_html(bin_file, file_label='File'):
            with open(bin_file, 'rb') as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
            return href
        st.markdown(get_binary_file_downloader_html('R0_sim.xlsx', 'R0_simp Data'), unsafe_allow_html=True)
except:
    print("Something went wrong")

#st.write("#### :globe_with_meridians: Country : **{}**".format(str(indi)))
st.subheader("📈 Evolution of R0 using Bettencourt & Rebeiro Method in "+select1)

try:
    df.to_excel("Data_covid.xlsx")
    url = 'Data_covid.xlsx'
    dg = pd.read_excel(url,
                        usecols=['Date', 'Country/Region', 'Difference'],
                        index_col=[1,0],
                        squeeze=True).sort_index()
    country_name = select1
    # We create an array for every possible value of Rt
    R_T_MAX = 12
    r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

    def prepare_cases(cases, cutoff=25):
        new_cases = cases

        smoothed = new_cases.rolling(7,
                win_type='gaussian',
                min_periods=1,
                center=True).mean(std=2).round()

        idx_start = np.searchsorted(smoothed, cutoff)

        smoothed = smoothed.iloc[idx_start:]
        original = new_cases.loc[smoothed.index]

        return original, smoothed
    cases = dg.xs(country_name).rename(f"{country_name} cases")

    original, smoothed = prepare_cases(cases)
    GAMMA = 1/7
    def get_posteriors(sr, sigma=0.15):

        # (1) Calculate Lambda
        lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))


        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
                data = sps.poisson.pmf(sr[1:].values, lam),
                index = r_t_range,
                columns = sr.index[1:])

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=r_t_range,
                                    scale=sigma
                                    ).pdf(r_t_range[:, None])

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior
        #prior0 = sps.gamma(a=4).pdf(r_t_range)
        prior0 = np.ones_like(r_t_range)/len(r_t_range)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
                index=r_t_range,
                columns=sr.index,
                data={sr.index[0]: prior0}
            )

        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        return posteriors, log_likelihood

    # Note that we're fixing sigma to a value just for the example
    posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
    def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
        if(isinstance(pmf, pd.DataFrame)):
            return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                                index=pmf.columns)

        cumsum = np.cumsum(pmf.values)

        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]

        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()

        # Find the smallest range (highest density)
        best = (highs - lows).argmin()

        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]

        return pd.Series([low, high],
                        index=[f'Low_{p*100:.0f}',
                                f'High_{p*100:.0f}'])

    hdi = highest_density_interval(posteriors, debug=True)
        # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=.9)

    most_likely = posteriors.idxmax().rename('ML')

    # Look into why you shift -1
    result = pd.concat([most_likely, hdis], axis=1)


    index = pd.to_datetime(result['ML'].index.get_level_values('Date'))
    values = result['ML'].values
    R0 = pd.DataFrame(list(zip(list(index), list(values))),
                columns =['Date', 'R0'])

    from_date  = st.selectbox('From :', list(R0.Date) )
    ind1=list(R0.Date).index(from_date)
    l2=list(R0.Date)[int(ind1)+1:]
    to_date= st.selectbox('To :',l2  )
    ind2=list(R0.Date).index(to_date)
    coun=[select1 for i in range(ind1,ind2+1)]
    R0_BR=list(values)[ind1:ind2+1]
    dt=list(R0.Date)[ind1:ind2+1]
    data_per=pd.DataFrame(list(zip(dt,R0_BR,coun)),
        columns =['Date',"R0","Country/Region"])
    fig = px.line(data_per,x="Date", y="R0")

    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="R0_Bettencourt&Rebeiro"
        )
    reference_line = go.Scatter(x=list(data_per['Date']),
                                y=[1 for i in range(len(list(data_per["Date"])))],
                                mode="lines",
                                line=go.scatter.Line(color="red"),

                                showlegend=False)
    fig.add_trace(reference_line)
    st.plotly_chart(fig)
    R0.index=R0.Date
    R0=data_per
    from scipy.signal import argrelextrema

    l=[R0["Date"][i].strftime("%d-%m-%Y") for i in range(len(R0))]
    R0["Date"]=l
    n=len(l)

    R0.plot(x="Date",y="R0",label="R0_Bettencourt_&_Ribeiro", figsize=(14,5), color="m")
    R0.to_excel("Data_Bettencourt_&_Ribeiro_"+select1+".xlsx")

    peak_indexes = argrelextrema(np.array(list(R0['R0'])), np.greater)
    peak_indexes = peak_indexes[0]
    plt.axhline(y=1,linestyle='--', color='black')

    # Plot peaks.
    peak_x = peak_indexes
    peak_y = np.array(list(R0['R0']))[peak_indexes]

    # Find valleys(min).
    valley_indexes = argrelextrema(np.array(list(R0['R0'])), np.less)
    valley_indexes = valley_indexes[0]


    # Plot valleys.
    valley_x = valley_indexes
    valley_y =  np.array(list(R0['R0']))[valley_indexes]

    reg_x=np.union1d(valley_indexes,peak_indexes)
    reg_y=np.array(list(R0['R0']))[reg_x]
    plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="Régression Linéaire")
    # Save graph to file.
    plt.xlabel('Date')
    plt.legend(loc='best')
    path=os.path.abspath(os.getcwd())
    plt.savefig(path+'\\R0_B&R.jpg')


    image = Image.open(path+'\\R0_B&R.jpg')
    st.image(image, caption='R0_Bettencourt & Rebeiro '+select1,
            use_column_width=True)
    #Downoad data

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html("Data_Bettencourt_&_Ribeiro_"+select1+".xlsx", 'R0_Bettencourt& Rebeiro Data'), unsafe_allow_html=True)
except:
    print("Something went wrong")

#Closure Time
st.subheader("📆 Closure Time in "+select1)
try:
    data = pd.DataFrame(df, columns=['Date','Confirmed',"Closure"]).set_index('Date')
    data['Date']=data.index

    date1=[]
    date2=[]
    for i in range(30,len(data)):
        a=data["Closure"][i]
        date1.append(data.index[i])
        b=data["Confirmed"][i]
        j=i
        while(b!=a and a<b):
            j=j-1
            b=data["Confirmed"][j]

        date2.append(data.index[j])
    days=[]
    for i in range(len(data)-30):
        days.append((date1[i]-date2[i]).days)
    l=[]
    for i in range(30):
        l.append(np.nan)
    l.extend(days)
    data["closure_time"]=l

    data['Country']=[select1 for i in range(len(l))]
    data.to_excel("closure_time.xlsx")
    fig = px.bar(data,x="Date", y="closure_time")

    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="Closure Time"
        )
    st.plotly_chart(fig)
    #Downoad data

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html("closure_time.xlsx", 'Closure Time Data'), unsafe_allow_html=True)
except:
    print("Something went wrong")

#IRR:

try:
    def irr(d1,dn,c1,cn):
        irr=(cn/c1)**(1/(dn-d1).days)-1

        return(irr)
    def DT(irr):
        p=m.log(2)/m.log(1+irr)
        return(p)
    df=data_cov[data_cov['Country/Region']==select1]
    data = pd.DataFrame(df, columns=['Date',"Country/Region",'Confirmed',"Closure"]).set_index('Date')
    n=list(data.index).index("02/03/2020")+30


    x=list(df['Date'])
    dates = [datetime.strptime(i1,'%d/%m/%Y') for i1 in x]

    data["date"]=dates
    IRR=[]
    DoubT=[]
    c1=data["Confirmed"][n]
    d1=dates[n]
    for i in range(n+1,len(df)):
        cn=data["Confirmed"][i]
        dn=dates[i]
        IRR.append(irr(d1,dn,c1,cn)*100)
        DoubT.append(DT(irr(d1,dn,c1,cn)))
    IRRn=[]
    for i in range(n+1):
        IRRn.append(np.nan)
    IRRn.extend(IRR)
    DoublingTime=[]
    for i in range(n+1):
        DoublingTime.append(np.nan)
    DoublingTime.extend(DoubT)
    data1=data.copy()
    data["IRR"]=IRRn
    data1['Doubling Time']=DoublingTime
    data.index=[i for i in range(len(data))]
    data1.index=[i for i in range(len(data1))]
    #Linear function function: f(x) = a*X+b

    def linear_f(X, a, b):
        y =a*X+b
        return y
# optimize from scipy


    linear_model, cov = optimize.curve_fit(linear_f,
                                xdata=np.arange(len(data1["Doubling Time"])-n-1),
                                ydata=data1["Doubling Time"].values[n+1:],
                                maxfev=10000,
                                p0=[np.max(list(data1["Doubling Time"])[n+1:]), 1])


    #linear function: f(x) = a*x+b

    def f(x):
        return linear_model[0]*x+linear_model[1]
    y_linear = f(x=np.arange(len(data1)-n-1))

    data1.to_excel("Doubling_Time.xlsx")
    data.to_excel("IRR.xlsx")
    a=str(round(linear_model[0],3))
    b=str(round(linear_model[1],3))
    x_values = data1["Doubling Time"].values[n+1:]
    y_values = y_linear
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    r2="(R²)= "+ str(round(r_squared,4))
    st.subheader("📆 Doubling Time in "+select1)
    fig = px.line(data1,x="date", y='Doubling Time')
    reference_line = go.Scatter(x=list(data1["date"])[n+1:],
                                y=y_linear,
                                mode="lines",
                                line=go.scatter.Line(color="red"),
                                name="Linear Regression",
                                showlegend=True)
    fig.add_trace(reference_line)

    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="Doubling Time"
        )
    st.write("Linear Regression: f(Time) = "+a+"*X +"+b)
    st.write("R-squared "+r2)
    st.plotly_chart(fig)
    #Downoad data

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html("Doubling_Time.xlsx", 'Doubling Time Data'), unsafe_allow_html=True)

    st.subheader("📉 Internal Rate of Return (IRR) in "+select1)
    fig = px.line(data,x="date", y='IRR')


    fig.update_layout(

            xaxis_title="Date",
            yaxis_title="IRR"
        )

    st.plotly_chart(fig)
    #Downoad data

    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html("IRR.xlsx", 'IRR Data'), unsafe_allow_html=True)
except:
    print("Something went wrong")
# Create a list of possible values and multiselect menu with them in it.

st.subheader("📝 Government Measures Taken in Response to COVID-19 in "+select1)
try:
    data=pd.read_excel("E:/hp/Downloads/measure1.xlsx")
    dfm=data[data["COUNTRY"]==select1]
    data_mar=dfm.rename(columns={'DATE_IMPLEMENTED':'Date'})
    groups1=list(data_mar["CATEGORY"])
    groups2=list(data_mar["_MEASURE"])
    gp=[]
    for i in range(len(groups1)):
        if [groups1[i],groups2[i]] not in gp:
            gp.append([groups1[i],groups2[i]])

    Category = data_mar['CATEGORY'].unique()
    gk=data_mar.groupby(['CATEGORY',"_MEASURE"])

    cat = st.selectbox('Select category', Category)
    l=[]
    for i in gp:
        if i[0]==cat:
            l.append(i[1])
    mesu = st.selectbox('Select Measure', l)
    dg=gk.get_group((cat,mesu))
    df=pd.read_excel("data.xlsx")
    dd=df.set_index('Date').join(dg.set_index('Date'))
    dd['Date']=dd.index
    mes=list(dd['_MEASURE'])

    fig = px.bar(dd,x="Date", y="Difference",text='COMMENTS'

                )
    j=0
    com=[]
    for i in range(len(mes)):
        #if math.isnan(mes[i])==True:
            #continue
        if type(mes[i])==str:
            j=j+1
            fig.add_annotation(


                x=list(dd['Date'])[i],
                text=str(j),
                y=list(dd['Difference'])[i]


            )
            com.append(str(j)+": "+list(dd['COMMENTS'])[i])


    fig.update_annotations(dict(
                xref="x",
                yref="y",


                arrowhead=7,
                ax=20,
                ay=-30,
                showarrow=True,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="center",

            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",

            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=2,
            bgcolor="#020a24",
            opacity=0.9
        ))

    st.plotly_chart(fig)
    for i in range(len(com)):
        st.markdown('**Measure ** {}  \n'.format(com[i]))
except:
    print("Something went wrong")