import streamlit as st
import pandas as pd
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
import wget
import os
import sqlite3
import pandas as pd
import wget
import os
import datetime





#END

image = Image.open("logo.jpg")
st.image(image,
		use_column_width=True)

#def cov():
def get_binary_file_downloader_html(bin_file, file_label='File'):
	with open(bin_file, 'rb') as f:
		data = f.read()
	bin_str = base64.b64encode(data).decode()
	href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
	return href

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstables(fn text,ln text,username TEXT,password TEXT);')


def add_userdata(fn,ln,username,password):
	c.execute('INSERT INTO userstables(fn,ln,username,password) VALUES (?,?,?,?)',(fn,ln,username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstables WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstables')
	data = c.fetchall()
	return data



def main():


	menu = ["Home","Log In","Sign Up"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.markdown('')
		st.title("Coronavirus disease:")
		st.markdown("""     """)
		st.markdown("""     """)
		st.markdown("""     """)
		image = Image.open("covid.jpg")
		st.image(image,
				use_column_width=True)
		st.markdown("""

	Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.

	Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.""")
		st.markdown("**Most common symptoms:**")
		st.markdown("""   ➛ Fever""")
		st.markdown("""   ➛ Dry cough""")
		st.markdown("""   ➛ Tiredness""")
		st.markdown("**Less common symptoms:**")
		st.markdown("   ➛ Aches and pains")
		st.markdown("""  \t➛ Sore throat""")
		st.markdown("""   ➛ Diarrhoea""")
		st.markdown("""   ➛ Conjunctivitis""")
		st.markdown("""   ➛ Headache""")
		st.markdown("""   ➛ Loss of taste or smell""")
		st.markdown("""   ➛ A rash on skin, or discolouration of fingers or toes""")
		st.markdown("**Serious symptoms:**")
		st.markdown("""   ➛ Difficulty breathing or shortness of breath""")
		st.markdown("""   ➛ Chest pain or pressure""")
		st.markdown("""   ➛ Loss of speech or movement""")
		st.markdown("""


		Seek immediate medical attention if you have serious symptoms. Always call before visiting your doctor or health facility.
		People with mild symptoms who are otherwise healthy should manage their symptoms at home.
		On average it takes 5–6 days from when someone is infected with the virus for symptoms to show, however it can take up to 14 days.""")

		st.markdown("""The best way to prevent and slow down transmission is to be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face.
			The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow).""")


	elif choice == "Log In":
		s="Login Section"
		st.sidebar.subheader(s)

		def generate_login_block():
			block1 = st.sidebar.empty()
			block2 = st.sidebar.empty()

			return block1, block2


		def clean_blocks(blocks):
			for block in blocks:
				block.empty()


		def login(blocks):
			blocks[0].markdown("""
					<style>
						input {
							-webkit-text-security: disc;
						}
					</style>
				""", unsafe_allow_html=True)
			blocks[1].markdown("""
					<style>
						input {
							-webkit-text-security: disc;
						}
					</style>
				""", unsafe_allow_html=True)

			return [blocks[0].text_input('Username',key="1"),blocks[1].text_input('Password',type='password',key="pas")]
		login_blocks = generate_login_block()
		Username,password=login(login_blocks)


		if st.sidebar.checkbox("Log In / Log Out"):
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)
			hash=check_hashes(password,hashed_pswd)

			result = login_user(Username,hash)
			if result:

				#st.success("Logged In as {}".format(Username))

				#clean_blocks(login_blocks)
				st.markdown("")
				st.markdown("")
				st.markdown("")
				st.markdown("")
				#st.title("Coronavirus disease:")
				dg=pd.read_excel("Covid.xlsx")
				from datetime import date

				today = date.today()



				# dd/mm/YY
				d1 = today.strftime("%d/%m/%Y")
				d2=list(dg['Date'])[-1]
				from datetime import datetime

				def days_between(d1, d2):
					d1 = datetime.strptime(d1, "%d/%m/%Y")
					d2 = datetime.strptime(d2, "%d/%m/%Y")
					return abs((d2 - d1).days)
				if d1!=list(dg['Date'])[-1] and days_between(d1, d2)>2 :
					url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv'
					url1 = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv'
					url2 = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv'
					wget.download(url, 'time_series_covid19_confirmed_global.csv')
					wget.download(url1, 'time_series_covid19_deaths_global.csv')
					wget.download(url2, 'time_series_covid19_recovered_global.csv')



					#Confirmed
					df = pd.read_csv('time_series_covid19_confirmed_global.csv')
					#print(df.columns)
					df = df.melt(id_vars = ['Province/State','Country/Region','Lat','Long'], var_name = None, value_name = 'Confirmed')

					df=df.drop(columns=['Province/State','Lat','Long'])

					#df = df.sum(level=['Country/Region','variable'])
					#df['e'] = df['Country/Region','variable'].sum(axis=0)

					df = df.replace('Western Sahara', 'Morocco')
					df = df.groupby(['Country/Region','variable'], as_index=False).agg({'Confirmed':'sum'})
					#print(df)

					#writer = pd.ExcelWriter('Confirmed.xlsx')
					#df.to_excel(writer)
					#writer.save()

					#Deaths
					df1 = pd.read_csv('time_series_covid19_deaths_global.csv')
					#print(df.columns)
					df1 = df1.melt(id_vars = ['Province/State','Country/Region','Lat','Long'], var_name = None, value_name = 'Deaths')

					df1 = df1.drop(columns=['Province/State','Lat','Long'])

					#df = df.sum(level=['Country/Region','variable'])
					#df['e'] = df['Country/Region','variable'].sum(axis=0)

					df1 = df1.replace('Western Sahara', 'Morocco')
					df1 = df1.groupby(['Country/Region','variable'], as_index=False).agg({'Deaths':'sum'})
					#print(df)

					#writer = pd.ExcelWriter('Deaths.xlsx')
					#df1.to_excel(writer)
					#writer.save()

					#Recovered
					df2 = pd.read_csv('time_series_covid19_recovered_global.csv')
					#print(df.columns)
					df2 = df2.melt(id_vars = ['Province/State','Country/Region','Lat','Long'], var_name = None, value_name = 'Recovered')

					df2 = df2.drop(columns=['Province/State','Lat','Long'])

					#df = df.sum(level=['Country/Region','variable'])
					#df['e'] = df['Country/Region','variable'].sum(axis=0)

					df2 = df2.replace('Western Sahara', 'Morocco')
					df2 = df2.groupby(['Country/Region','variable'], as_index=False).agg({'Recovered':'sum'})
					#print(df)

					#Join
					#new_df = pd.merge(df, df1,  how='left', left_on=['Country/Region','variable'], right_on = ['Country/Region','variable'])
					df = df.join(df1.set_index(['Country/Region','variable']), on=['Country/Region','variable'])
					df = df.join(df2.set_index(['Country/Region','variable']), on=['Country/Region','variable'])
					#print(df)

					#Change date type
					df['variable'] = pd.to_datetime(df['variable'])
					#df['variable'] = df['variable'].dt.strftime('%d/%m/%Y')

					#print(df)

					#Remove zeros
					indexNames = df[ (df['Confirmed'] == 0)
									& (df['Deaths'] == 0)
									& (df['Recovered'] == 0)].index
					df.drop(indexNames , inplace=True)
					#print(df)

					print('Finish importing')

					#Rename columns
					df.rename(columns={'variable': 'Date'}, inplace=True)
					df = df.reindex(columns=['Date','Country/Region','Confirmed','Deaths','Recovered'])
					df = df.sort_values(by=['Country/Region', 'Date'])

					#Calculate new columns
					df['Actif'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
					df['Closure'] = df['Recovered'] + df['Deaths']
					df['Difference'] = df.groupby('Country/Region')['Confirmed'].diff()
					df['Difference'] = df['Difference'].fillna(0)

					dff=df['Date']

					#Format date
					df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')



					print('Exporting Excel...')
					#Write excel
					writer = pd.ExcelWriter('Covid.xlsx',date_format='dd/mm/yyyy')
					df.to_excel(writer,index=False)
					writer.save()
					print('Excel Saved')
					os.remove("time_series_covid19_confirmed_global.csv")
					os.remove("time_series_covid19_recovered_global.csv")
					os.remove("time_series_covid19_deaths_global.csv")
					df.to_excel("Covid.xlsx",index=False)
					dg=pd.read_excel("Covid.xlsx")
					data_cov=dg
				else:
					data_cov=dg
				data_cov1=data_cov.copy()
				#Country/Region
				c=list((data_cov["Country/Region"]))
				c=list(np.unique(c))

				select0= st.sidebar.selectbox('Select :', ["Overview","World Data","By country","Country comparison covid-19", "Somalia Coronavirus"], key='2')
				if select0=="Somalia Coronavirus":
					#dso=data_cov[data_cov["Country/Region"]=="Somalia"]
					st.markdown('Please select your file extension')
					ty=['XLSX','CSV']
					rad_types = st.radio(label="", options=ty)
					st.set_option('deprecation.showfileUploaderEncoding', False)
					uploaded_file = st.file_uploader("Choose a XLSX or CSV file", type=["xlsx","csv"])
					from pathlib import Path


					if uploaded_file:
						extension = Path(uploaded_file.name).suffix
						if extension.upper()==".XLSX":
							dso = pd.read_excel(uploaded_file)
						else:
							dso = pd.read_csv(uploaded_file)




					#dso=data_cov[data_cov["Country/Region"]=="Somalia"]
					#uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

					#if uploaded_file:
						#dso = pd.read_excel(uploaded_file)






						dates = list(dso["Date"])
						da= []
						for i in dates:
							da.append("{0:-08d}".format(i))


						d=[]
						for i in da:
							d.append(datetime.strptime(i, '%Y%m%d'))
						dso["Date"]=d
						du=list(np.unique(d))
						c=list((dso["STATE"]))
						num=(d[-1]-d[0]).days
						c=list(np.unique(c))
						all_cumul=[]
						all_day=[]
						reg_cumul=list(dso["Cumulative cases"])
						reg_day=list(dso["Cases by day"])
						cn=list(dso['Country'])[0]

						state=["All" for i in du]
						som=[ cn for i in du]
						index=[]
						for i in du:
							index=[j for j,val in enumerate(d) if val==i]
							a=0
							b=0
							for k in index:
								a=a+reg_cumul[k]
								b=b+reg_day[k]
							all_cumul.append(a)
							all_day.append(b)
						dataf=pd.DataFrame(list(zip(du,som,state,all_cumul,all_day)),columns =['Date',"Country","STATE","Cumulative cases","Cases by day"])
						dso=dso.append(dataf,ignore_index = True)
						c.append("All")
						select_reg= st.sidebar.selectbox('Select a region:',c, key='2')
						data_reg=dso[dso['STATE']==select_reg]
						rad=['Confirmed cases',"New cases per day","Logistic Model","R0 (simplistic Method)","R0 (Bettencourt and Rebeiro Method)","Doubling Time","IRR"]

						radio = st.sidebar.radio(label="", options=rad)
						if radio == "Confirmed cases":

							st.subheader("Confirmed cases:")
							st.markdown("")


							fig = go.Figure()
							fig.add_trace(go.Bar(name="Confirmed cases", x=list(data_reg['Date']), y=list(data_reg['Cumulative cases'])))
							fig.update_layout(
								showlegend=True,
								xaxis_title="Time (Date)",
								yaxis_title="Nº of cumulative cases",
								plot_bgcolor='white'

							)

							st.plotly_chart(fig)
							st.subheader('Data')
							st.write(data_reg)
							data_reg.to_excel("somalia_reg.xlsx")
							st.markdown(get_binary_file_downloader_html('somalia_reg.xlsx', 'Data'), unsafe_allow_html=True)
						elif radio=="IRR" or radio=="Doubling Time":
							try:

								df = data_reg





								def irr(d1,dn,c1,cn):
									irr=(cn/c1)**(1/(dn-d1).days)-1

									return(irr)
								def DT(irr):
									p=m.log(2)/m.log(1+irr)
									return(p)
								data = pd.DataFrame(df, columns=['Date',"STATE",'Cumulative cases']).set_index('Date')

								x=list(df['Date'])

								data["date"]=x
								dates = list(df["Date"])
								#st.write(dates[0])
								#y=list(r0['Date'])
								#dates = [datetime.strptime(i1,'%d-%m-%Y') for i1 in y]
								#st.write(dates[0])
								#r0["Date"]=dates

								#n1=list(data["date"]).index(list(dr['Date'])[0])
								#n2=list(data["date"]).index(list(dr['Date'])[-1])
								#st.write(n1)
								#st.write(n2)

								#data=data[n1:n2+1]
								IRR=[]
								DoubT=[]
								l=list(data["Cumulative cases"])
								#n=0
								#for i in range(len(l)-1):
									#if l[i]!=l[i+1]:
										#n=i
										#break
								#st.markdown(n)
								n=44



								c1=data["Cumulative cases"][n]
								d1=dates[n]
								for i in range(n+1,len(data)):
									cn=data["Cumulative cases"][i]
									dn=dates[i]
									if irr(d1,dn,c1,cn)*100==0:
										IRR.append(0)
										DoubT.append(0)
									else:

										IRR.append(irr(d1,dn,c1,cn)*100)
										DoubT.append(DT(irr(d1,dn,c1,cn)))
								IRRn=[]
								for i in range(n+1):
									IRRn.append(np.nan)
								IRRn.extend(IRR)

								#st.markdown(IRRn)
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
								if radio=="Doubling Time":
									st.subheader("📆 Doubling Time in "+select_reg)
									st.markdown("")
									st.markdown("**The doubling time of an epidemic **  is the period of time required for the number of cases in the epidemic to double.")
									fig= go.Figure()


									reference_line = go.Scatter(x=list(data1["date"]),
																y=list(data1["Doubling Time"]),
																mode="lines",
																line=go.scatter.Line(color="blue"),
																name="Doubling Time",
																showlegend=True)
									fig.add_trace(reference_line)
									reference_line = go.Scatter(x=list(data1["date"])[n+1:],
																y=y_linear,
																mode="lines",
																line=go.scatter.Line(color="red"),
																name="Linear Regression",
																showlegend=True)
									fig.add_trace(reference_line)

									fig.update_layout(

											xaxis_title="Date",
											yaxis_title="Doubling Time",
											plot_bgcolor='white'
										)
									st.write("Linear Regression: f(Time) = "+a+"*X +"+b)
									st.write("R-squared "+r2)
									st.plotly_chart(fig)
									#Downoad data


									st.markdown(get_binary_file_downloader_html("Doubling_Time.xlsx", 'Doubling Time Data'), unsafe_allow_html=True)
								else:
									st.subheader("📉 Incidence Rate Ratio (IRR) in "+select_reg)
									st.markdown('')
									st.markdown('Incidence measures the proportion of the population affected by the disease at a given time, it is one of the two most used indicators in epidemiology to assess the frequency and the spead of disease.')
									fig = go.Figure()
									fig.add_trace(go.Line(name="IRR ", x=list(data['date']), y=list(data['IRR'])))
									fig.update_layout(
										showlegend=True,




											xaxis_title="Date",
											yaxis_title="IRR",
											plot_bgcolor='white'
										)

									st.plotly_chart(fig)
									#Downoad data


									st.markdown(get_binary_file_downloader_html("IRR.xlsx", 'IRR Data'), unsafe_allow_html=True)
							except:
								print("Something went wrong")
						elif radio=="R0 (simplistic Method)":


							try:
								st.subheader("📈 Evolution of R0 using Simplist Method in "+select_reg)
								st.markdown("")

								st.markdown("""R0, pronounced “R naught,” is a mathematical term that indicates how contagious an infectious disease is. It’s also referred to as the reproduction number. As an infection is transmitted to new people, it reproduces itself.""")

								st.markdown("""R0 tells you the average number of people who will contract a contagious disease from one person with that disease. It specifically applies to a population of people who were previously free of infection and haven’t been vaccinated.""")

								st.markdown("""For example, if a disease has an R0 of 18, a person who has the disease will transmit it to an average of 18 other people. That replication will continue if no one has been vaccinated against the disease or is already immune to it in their community.""")
								image = Image.open("r0.png")
								st.image(image,
								use_column_width=True)
								st.markdown("***Period:***")
								data = pd.DataFrame(df, columns=['Date','Difference']).set_index('Date')

								data['smooth_mean(gauss, win=7)'] = data.iloc[:,0].rolling(7,
									win_type='gaussian',
									min_periods=1,
									center=True).mean(std=2).round()
								ds=list(data['smooth_mean(gauss, win=7)'])
								for i in range(len(ds)):
									if ds[i]==0:
										ds[i]=np.nan
								data['smooth_mean(gauss, win=7)']=ds
								gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
								l_7=[]
								for i in range(len(gauss)-6):
									if  m.isnan(gauss[i])==True or m.isnan(gauss[i+6])==True:
										l_7.append(np.nan)
									else:
										l_7.append(round(gauss[i+6])/round(gauss[i]))

								gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
								l_4=[]
								for i in range(len(gauss)-3):
									if  m.isnan(gauss[i])==True or m.isnan(gauss[i+3])==True :
										l_4.append(np.nan)
									else:
										l_4.append(round(gauss[i+3])/round(gauss[i]))
								N=len(data)
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
								data["Gaussian_R0_4_Days"]=gauss_4
								data["Gaussian_R0_7_Days"]=gauss_7
								col = data.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
								data['R0_Simpliste'] = col.mean(axis=1)
								R0_sim=list(data['R0_Simpliste'])
								data['Date']=data.index
								from_date  = st.selectbox('From :', list(data.Date) )
								ind1=list(data.Date).index(from_date)
								l2=list(data.Date)[int(ind1)+1:]
								to_date= st.selectbox('To :',l2  )
								ind2=list(data.Date).index(to_date)
								R0_sim=R0_sim[ind1:ind2+1]
								dt=list(data.Date)[ind1:ind2+1]
								data_per=pd.DataFrame(list(zip(dt,R0_sim)),
									columns =['Date',"R0_Simpliste"])
								fig = go.Figure()
								fig.add_trace(go.Line(name="R0 - Simplistic Method", x=list(data_per['Date']), y=list(data_per['R0_Simpliste'])))
								fig.update_layout(
								showlegend=True,

								plot_bgcolor='white',
								xaxis_title="Date",
								yaxis_title="R0 Simplistic"
							)
								reference_line = go.Scatter(x=list(data_per['Date']),
													y=[1 for i in range(len(dt))],
													mode="lines",
													line=go.scatter.Line(color="red"),

													showlegend=False)
								fig.add_trace(reference_line)
								st.plotly_chart(fig)

								from scipy.signal import argrelextrema
								data=data_per
								data['Country']=[select_reg for i in range(len(data))]
								da=[data['Date'][i].strftime("%d-%m-%Y") for i in range(len(data))]
								data['Date']=da
								data.plot(x="Date",y="R0_Simpliste",label="R0", figsize=(14,5), color="m")


								peak_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.greater)
								peak_indexes = peak_indexes[0]
								plt.axhline(y=1,linestyle='--', color='black')

								# Plot peaks.
								peak_x = peak_indexes
								peak_y = np.array(list(data['R0_Simpliste']))[peak_indexes]

								# Find valleys(min).
								valley_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.less)
								valley_indexes = valley_indexes[0]


								# Plot valleys.
								valley_x = valley_indexes
								valley_y =  np.array(list(data['R0_Simpliste']))[valley_indexes]

								reg_x=np.union1d(valley_indexes,peak_indexes)
								reg_y=np.array(list(data['R0_Simpliste']))[reg_x]
								plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="linear regression")
								# Save graph to file.
								plt.xlabel('Date')
								plt.legend(loc='best')

								plt.title("R0_Simpliste "+select_reg)
								plt.legend(loc='best')
								#path=os.path.abspath(os.getcwd())
								plt.savefig('R0_Sim.jpg')
								image = Image.open('R0_Sim.jpg')
								st.image(image, caption='R0_Simplist '+select_reg,
								use_column_width=True)

								#Downoad data
								st.markdown("""""")
								st.markdown("""**Note: ** World knows both a decrease in the number of new reported cases and an increase in it. In fact, some countries sometimes report zero coronavirus cases for a period of time as China, Somalia... This variance can influence the calculation of R0. That's why you can observe some missing values.    """)

								data_per.to_excel("R0_sim.xlsx")


								st.markdown(get_binary_file_downloader_html('R0_sim.xlsx', 'R0_simp Data'), unsafe_allow_html=True)
							except:
								print("Something went wrong")

						elif radio=="R0 (Bettencourt and Rebeiro Method)":

							#st.write("#### :globe_with_meridians: Country : **{}**".format(str(indi)))


							try:
								st.subheader("📈 Evolution of R0 using Bettencourt & Rebeiro Method in "+select_reg)
								st.markdown("")

								st.markdown("""R0, pronounced “R naught,” is a mathematical term that indicates how contagious an infectious disease is. It’s also referred to as the reproduction number. As an infection is transmitted to new people, it reproduces itself.""")

								st.markdown("""R0 tells you the average number of people who will contract a contagious disease from one person with that disease. It specifically applies to a population of people who were previously free of infection and haven’t been vaccinated.""")

								st.markdown("""For example, if a disease has an R0 of 18, a person who has the disease will transmit it to an average of 18 other people. That replication will continue if no one has been vaccinated against the disease or is already immune to it in their community.""")
								image = Image.open("r0.png")
								st.image(image,
								use_column_width=True)
								st.markdown("***R0:***")

								df=data_reg

								df.to_excel("Data_covid.xlsx")
								url = 'Data_covid.xlsx'
								df = pd.read_excel(url,
													usecols=['Date', 'STATE', 'Cases by day'],
													index_col=[1,0],
													squeeze=True).sort_index()
								country_name = select_reg
								# We create an array for every possible value of Rt
								R_T_MAX = 12
								r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

								def prepare_cases(cases, cutoff=2):
									new_cases = cases

									smoothed = new_cases.rolling(7,
										win_type='gaussian',
										min_periods=1,
										center=True).mean(std=2).round()

									idx_start = np.searchsorted(smoothed, cutoff)

									smoothed = smoothed.iloc[idx_start:]
									original = new_cases.loc[smoothed.index]

									return original, smoothed
								cases = df.xs(country_name).rename(f"{country_name} cases")

								original, smoothed = prepare_cases(cases,5)
								l=[i for i in range(len(smoothed)) if smoothed[i]==0]
								for i in l:
									smoothed[i]=1

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
										if denominator!=0:

									# Execute full Bayes' Rule
											posteriors[current_day] = numerator/denominator

									# Add to the running sum of log likelihoods

											log_likelihood += np.log(denominator)

									return posteriors, log_likelihood

								# Note that we're fixing sigma to a value just for the example
								posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
								posteriors=posteriors.dropna(axis=1, how='all')
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
								#result0 = pd.concat([most_likely, hdis], axis=1)
								#original, smoothed = prepare_cases(cases,10)
								#posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
								#posteriors=posteriors.dropna(axis=1, how='all')
								#hdi = highest_density_interval(posteriors, debug=True)
								# Note that this takes a while to execute - it's not the most efficient algorithm
								#hdis = highest_density_interval(posteriors, p=.9)

								#most_likely = posteriors.idxmax().rename('ML')
								result = pd.concat([most_likely, hdis], axis=1)
								#if result1.index[-1]==result0.index[-1]:
									#result=result0
								#else:
									#jours=(result1.index[0]-result0.index[-1]).days
									#st.markdown(jours)
									#if  jours==0:
										#result=result0
									#else:
										#res=result0
										#ld=list(result0.index)
										#lr=list(result0['ML'])
										#for i in range((result1.index[0]-result0.index[-1]).days):
											#ld.append(result0.index[-1]+timedelta(days=i+1))
											#lr.append(np.nan)

										#df= pd.DataFrame(list(zip(ld, lr)),
													#columns =['Date', 'ML'])
										#re = pd.DataFrame(df, columns=['Date',"ML"]).set_index('Date')
										#result=re.append(result1)



								index = pd.to_datetime(result['ML'].index.get_level_values('Date'))
								values = result['ML'].values
								R0 = pd.DataFrame(list(zip(list(index), list(values))),
											columns =['Date', 'R0'])
								R0.index=R0.Date
								r0=R0.copy()

								coun=[select_reg for i in range(len(list(values)))]
								R0_BR=list(values)
								dt=list(result.index)
								data_per=pd.DataFrame(list(zip(dt,R0_BR,coun)),
									columns =['Date',"R0","STATE"])
								fig = go.Figure()
								fig.add_trace(go.Line(name="R0 - Bettencourt & Rebeiro ", x=list(data_per['Date']), y=list(data_per['R0'])))
								fig.update_layout(
								showlegend=True,


										xaxis_title="Date",
										yaxis_title="R0_Bettencourt & Rebeiro",
										plot_bgcolor='white'
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
								R0.to_excel("Data_Bettencourt_&_Ribeiro_"+select_reg+".xlsx")

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
								#path=os.path.abspath(os.getcwd())
								plt.savefig('R0_B&R.jpg')


								image = Image.open('R0_B&R.jpg')
								st.image(image, caption='R0_Bettencourt & Rebeiro '+select_reg,
										use_column_width=True)
								st.markdown("""""")
								st.markdown("""**Note: ** World knows both a decrease in the number of new reported cases and an increase in it. In fact, some countries sometimes report zero coronavirus cases for a period of time as China, Somalia... This variance can influence the calculation of R0. That's why you can observe some missing values.    """)
								#Downoad data


								st.markdown(get_binary_file_downloader_html("Data_Bettencourt_&_Ribeiro_"+select_reg+".xlsx", 'R0_Bettencourt& Rebeiro Data'), unsafe_allow_html=True)

							except:

								st.markdown("Sorry, something went wrong you can visualize R0 using Simplistic Method. ")
							#st.write(type(list(data_per['Date'])[0]))
						if radio == "New cases per day":

							st.markdown("")
							st.subheader("New cases:")
							data = pd.DataFrame(data_reg, columns=['Date','Cases by day']).set_index('Date')


							data['Smoothed'] = data.iloc[:,0].rolling(7,
									win_type='gaussian',
									min_periods=1,
									center=True).mean(std=2).round()



							fig = go.Figure()
							fig.add_trace(go.Bar(name="Cases by day", x=list(data.index), y=list(data['Cases by day'])))
							fig.add_trace(go.Line(name="Time serie of infection (Smoothed)", x=list(data.index), y=list(data['Smoothed'])))
							data.to_excel("data.xlsx")

							fig.update_layout(
								showlegend=True,

								xaxis_title="Time (Date)",
								yaxis_title="Nº of Cases by day",
								plot_bgcolor='white'

							)

							st.plotly_chart(fig)
							st.subheader('Data')
							st.write(data)
							st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
						elif radio=="R0 (simplistic Method)":


							try:
								st.subheader("📈 Evolution of R0 using Simplist Method in "+select_reg)
								st.markdown("")

								st.markdown("""R0, pronounced “R naught,” is a mathematical term that indicates how contagious an infectious disease is. It’s also referred to as the reproduction number. As an infection is transmitted to new people, it reproduces itself.""")

								st.markdown("""R0 tells you the average number of people who will contract a contagious disease from one person with that disease. It specifically applies to a population of people who were previously free of infection and haven’t been vaccinated.""")

								st.markdown("""For example, if a disease has an R0 of 18, a person who has the disease will transmit it to an average of 18 other people. That replication will continue if no one has been vaccinated against the disease or is already immune to it in their community.""")
								image = Image.open("r0.png")
								st.image(image,
								use_column_width=True)
								st.markdown("***Period:***")
								df=data_reg
								data = pd.DataFrame(df, columns=['Date','Cases by day']).set_index('Date')

								data['smooth_mean(gauss, win=7)'] = data.iloc[:,0].rolling(7,
									win_type='gaussian',
									min_periods=1,
									center=True).mean(std=2).round()
								ds=list(data['smooth_mean(gauss, win=7)'])
								for i in range(len(ds)):
									if ds[i]==0:
										ds[i]=np.nan
								data['smooth_mean(gauss, win=7)']=ds
								gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
								l_7=[]
								for i in range(len(gauss)-6):
									if  m.isnan(gauss[i])==True or m.isnan(gauss[i+6])==True:
										l_7.append(np.nan)
									else:
										l_7.append(round(gauss[i+6])/round(gauss[i]))

								gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
								l_4=[]
								for i in range(len(gauss)-3):
									if  m.isnan(gauss[i])==True or m.isnan(gauss[i+3])==True :
										l_4.append(np.nan)
									else:
										l_4.append(round(gauss[i+3])/round(gauss[i]))
								N=len(data)
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
								data["Gaussian_R0_4_Days"]=gauss_4
								data["Gaussian_R0_7_Days"]=gauss_7
								col = data.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
								data['R0_Simpliste'] = col.mean(axis=1)
								R0_sim=list(data['R0_Simpliste'])
								data['Date']=data.index
								from_date  = st.selectbox('From :', list(data.Date) )
								ind1=list(data.Date).index(from_date)
								l2=list(data.Date)[int(ind1)+1:]
								to_date= st.selectbox('To :',l2  )
								ind2=list(data.Date).index(to_date)
								R0_sim=R0_sim[ind1:ind2+1]
								dt=list(data.Date)[ind1:ind2+1]
								data_per=pd.DataFrame(list(zip(dt,R0_sim)),
									columns =['Date',"R0_Simpliste"])
								fig = go.Figure()
								fig.add_trace(go.Line(name="R0 - Simplistic Method", x=list(data_per['Date']), y=list(data_per['R0_Simpliste'])))
								fig.update_layout(
								showlegend=True,

								plot_bgcolor='white',
								xaxis_title="Date",
								yaxis_title="R0 Simplistic"
							)
								reference_line = go.Scatter(x=list(data_per['Date']),
													y=[1 for i in range(len(dt))],
													mode="lines",
													line=go.scatter.Line(color="red"),

													showlegend=False)
								fig.add_trace(reference_line)
								st.plotly_chart(fig)

								from scipy.signal import argrelextrema
								data=data_per
								data['Country']=[select_reg for i in range(len(data))]
								da=[data['Date'][i].strftime("%d-%m-%Y") for i in range(len(data))]
								data['Date']=da
								data.plot(x="Date",y="R0_Simpliste",label="R0", figsize=(14,5), color="m")


								peak_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.greater)
								peak_indexes = peak_indexes[0]
								plt.axhline(y=1,linestyle='--', color='black')

								# Plot peaks.
								peak_x = peak_indexes
								peak_y = np.array(list(data['R0_Simpliste']))[peak_indexes]

								# Find valleys(min).
								valley_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.less)
								valley_indexes = valley_indexes[0]


								# Plot valleys.
								valley_x = valley_indexes
								valley_y =  np.array(list(data['R0_Simpliste']))[valley_indexes]

								reg_x=np.union1d(valley_indexes,peak_indexes)
								reg_y=np.array(list(data['R0_Simpliste']))[reg_x]
								plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="linear regression")
								# Save graph to file.
								plt.xlabel('Date')
								plt.legend(loc='best')

								plt.title("R0_Simpliste "+select_reg)
								plt.legend(loc='best')
								#path=os.path.abspath(os.getcwd())
								plt.savefig('R0_Sim.jpg')
								image = Image.open('R0_Sim.jpg')
								st.image(image, caption='R0_Simplist '+select_reg,
								use_column_width=True)

								#Downoad data
								st.markdown("""""")
								st.markdown("""**Note: ** World knows both a decrease in the number of new reported cases and an increase in it. In fact, some countries sometimes report zero coronavirus cases for a period of time as China, Somalia... This variance can influence the calculation of R0. That's why you can observe some missing values.    """)

								data_per.to_excel("R0_sim.xlsx")


								st.markdown(get_binary_file_downloader_html('R0_sim.xlsx', 'R0_simp Data'), unsafe_allow_html=True)
							except:
								print("Something went wrong")


						elif radio=="Logistic Model":

							#try:
								df=data_reg
								df.index=[i for i in range(len(df))]

								x = df["Cumulative cases"].index
								y =df["Cumulative cases"].values

								# inflection point estimation
								inf="no"
								fl=0
								while(inf=="no"):
									try:
										dy = np.diff(y,n=1+fl) #  derivative
										idx_max_dy = np.argmax(dy)

										#Logistic function: f(x) = capacity / (1 + e^-k*(x - midpoint) )

										def logistic_f1(X, c, k, m):
											y = c / (1 + np.exp(-k*(X-m)))
											return y
										# optimize from scipy


										logistic_model1, cov = optimize.curve_fit(logistic_f1,
																		xdata=np.arange(len(df["Cumulative cases"])+np.argmax(dy)-len(df)),
																		ydata=df["Cumulative cases"].values[0: np.argmax(dy)],
																		maxfev=10000,
																		p0=[np.max(list(df["Cumulative cases"])[0: np.argmax(dy)]), 1, 1])


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
																		xdata=np.arange(len(df["Cumulative cases"])- np.argmax(dy)),
																		ydata=df["Cumulative cases"].values[ np.argmax(dy):],
																		maxfev=10000,
																		p0=[np.max(list(df["Cumulative cases"])[ np.argmax(dy):]), 1, 1])

										#Logistic function: f(x) = a / (1 + e^(-b*(x-c)))

										def f(x):
											return logistic_model2[0] / (1 + np.exp(-logistic_model2[1]*(x-logistic_model2[2])))

										y_logistic2 = f(x=np.arange( len(df)-np.argmax(dy))) # 60 ==> Prédiction des cas confirmés dans les futurs 2 mois.
										inf="yes"
									except:
											inf="no"
											fl=fl+1
								k=0
								#st.markdown(y_logistic2)
								while int(y_logistic2[len(y_logistic2)-1]) - int(y_logistic2[len(y_logistic2)-2]) !=0 :
									k=k+1
									y_logistic2 = f(x=np.arange( len(df)-np.argmax(dy)+k))
								#st.markdown(k)
								confirm=list(df["Cumulative cases"])
								no=[np.nan for i in range(len(df),len(df)+k)]
								confirm.extend(no)
								index=[i for i in range(len(df)+k)]
								log=[]
								log.extend(y_logistic1)
								log.extend(y_logistic2)
								date=du
								#date=list(df['Date'])[0:len(y_logistic1)]
								cases=list(df["Cases by day"])
								#cases.append(abs(list(df["Cumulative cases"])[-1]-log[len(list(df["Cumulative cases"]))]))
								ll=list(np.diff(log[len(list(df["Cumulative cases"])):]))
								cases.extend(ll)

								fin=date[-1]
								#st.markdown(fin)
								for i in range(k):
									k1=fin+timedelta(days=i+1)

									date.append(k1)

								log_df = pd.DataFrame(list(zip(index,date, confirm,log)),
											columns =['index', "Date",'Confirmed',"Predicted"])
								#st.write(log_df)
								st.subheader("☞ Logistic Model ")
								st.markdown("")

								st.markdown("The models based on mathematical statistics, machine learning and deep learning have been applied to the prediction of time series of epidemic development. Logistic is often used in regression fitting of time series data due to its simple principle and efficient calculation. For example, in the Coronavirus case, Logistic growth is characterized by a slow increase in growth at the beginning, fast growth phase approaching the peak of the incidence curve, and a slow growth phase approaching the end of the outbreak (the maximum of infections).")
								st.markdown("**Logistic Function :**")
								image = Image.open("lf1.png")
								st.image(image,
								use_column_width=True)
								st.markdown("**Modeling "+ select_reg+" COVID-19 Cumulative Confirmed Cases:**")
								st.markdown("**An inflection point** is a point in a graph at which the concavity changes, it represents also an event that results in a significant change in the progress of a company, industry, sector, economy... and can be considered a turning point after which a dramatic change, with either positive or negative results, is expected to result.")
								st.markdown("To model the evolution of confirmed cases we use 2 logistic funtions:")
								#Logistic function: f(x) = a / (1 + e^(-b*(x-c)))
								a1=str(round(logistic_model1[0],2))
								b1=str(round(logistic_model1[1],2))
								c1=str(round(logistic_model1[2],2))
								st.markdown(" **+**  f1(x)="+a1+"/(1+e^(-"+b1+"*(x-"+c1+")))")

								a2=str(round(logistic_model2[0],2))
								b2=str(round(logistic_model2[1],2))
								c2=str(round(logistic_model2[2],2))
								st.markdown("  **+** f2(x)="+a2+"/(1+e^(-"+b2+"*(x-"+c2+")))")
								correlation_matrix = np.corrcoef(log[0:len(df)],list(log_df["Confirmed"])[0:len(df)] )
								correlation_xy = correlation_matrix[0,1]
								r_squared = correlation_xy**2
								r2="       **R²** = "+ str(round(r_squared*100,2))+"%"
								st.markdown(r2)





								fig = go.Figure()
								fig_case=go.Figure()

								fig2=go.Figure()
								reference_line = go.Scatter(x=list(log_df["Date"]),
															y= list(log_df["Confirmed"]),
															mode="lines",
															line=go.scatter.Line(color="blue"),
															name="Confimed cases",
															showlegend=True)
								fig.add_trace(reference_line)
								reference_line2 = go.Scatter(x=list(log_df["Date"])[0:len(df)],
															y= list(log_df["Confirmed"])[0:len(df)],
															mode="lines",
															line=go.scatter.Line(color="blue"),
															name="Confimed cases",
															showlegend=True)
								fig2.add_trace(reference_line2)


								reference_line = go.Scatter(x=date,
															y=log,
															mode="lines",
															line=go.scatter.Line(color="red"),
															name="Predicted",
															showlegend=True)
								fig.add_trace(reference_line)

								fig2.add_trace(go.Line(name="Logistic Function 1", x=date[0:len(y_logistic1)], y=y_logistic1))
								fig_case.add_trace(go.Bar(name="New cases per Day", x=date, y=cases))
								fig2.add_trace(go.Line(name="Logistic Function 2", x=date[len(y_logistic1):len(df)], y=y_logistic2[0:len(df)]))


								a1=list(log_df["Date"])[np.argmax(dy)]
								l1=[]
								l1.append(a1)
								a2=list(log_df["Confirmed"])[np.argmax(dy)]
								l2=[]
								l2.append(a2)

								fig2.add_trace(go.Line(name="Inflection point", x=l1,y=l2))

								fig.update_layout(
								showlegend=True,


										xaxis_title="Date",
										plot_bgcolor='white'
									)

								fig2.update_layout(
								showlegend=True,


										xaxis_title="Date",
										plot_bgcolor='white'
									)

								st.plotly_chart(fig2)


								fig_case.update_layout(
								showlegend=True,


										xaxis_title="Date",
										plot_bgcolor='white'
									)


								st.markdown("**Predict the end date of Covid-19 in "+select_reg+"**")
								st.markdown("The graphical representaion below shows the end date of covid-19 and the total number of confirmed cases in  "+select_reg)

								st.plotly_chart(fig)
								st.markdown('**The end date: **'+date[-1].strftime("%Y-%m-%d"))
								st.markdown('**The total number of confirmed cases : **'+str(int(log[-1])))
								st.markdown("")
								st.markdown("** 📊 The number of new cases per day **")
								st.markdown("We can extract and predict the number of new cases per day using logistic model:")
								st.plotly_chart(fig_case)

							#except:
								#print("Something went wrong")


				if select0=="World Data":
					st.markdown("""     """)
					image = Image.open("world.jpg")
					st.image(image,
								use_column_width=True)
					st.markdown("Covid-19 has spread around the planet, sending billions of people into lockdown as health services struggle to cope. It is affecting 214 countries and territories around the world and 2 international conveyances.")
					st.markdown("")
					st.markdown("**Data:**")
					st.write(data_cov)
					data_cov.to_excel("Covid.xlsx")
					st.markdown(get_binary_file_downloader_html('Covid.xlsx', 'Covid_data'), unsafe_allow_html=True)



				elif select0=="Country comparison covid-19":
					df=data_cov1
					l=list(df["Country/Region"])
					countries=list(np.unique(l))
					multiselection = st.multiselect("Select countries:", countries)
					df = df[df["Country/Region"].isin(multiselection)]
					dates = list(df["Date"])
					d=[]
					for i in dates:

						d.append(datetime.strptime(i, '%d/%m/%Y'))
					df['Date']=d
					rad1=["Lethality","Recovery","Closure Time","R0 - simplistic method","R0 - Bettencourt & Rebeiro Method","Doubling Time","IRR"]
					op = st.sidebar.radio(label="", options=rad1)
					count=multiselection
					if op=="Lethality":
						try:
							lethal=[]
							ld=[]
							for count in multiselection:

								data=df[df["Country/Region"]==count]
								clo=list(data['Closure'])
								death=list(data['Deaths'])
								conf=list(data['Confirmed'])
								n=len(clo)

								letal1=[]
								for i in range(30,n):
									letal1.append(100*(death[i]/(conf[i])))
								l1=[]
								for i in range(30):
									l1.append(np.nan)
								l1.extend(letal1)


								data["lethality"]=l1
								ld.append(data)
							df_row = pd.concat(ld)
							fig = px.line(df_row,x="Date", y="lethality",color="Country/Region")

							fig.update_layout(

									xaxis_title="Date",
									yaxis_title="Lethality",



									plot_bgcolor='white'

								)
							st.plotly_chart(fig)
							df_row.to_excel("lethality.xlsx")


							st.markdown(get_binary_file_downloader_html('lethality.xlsx', 'Lethality_Data'), unsafe_allow_html=True)



						except:
							print("Something went wrong")
					elif op=="Recovery":
						try:
							rec=[]
							ld=[]
							for count in multiselection:

								data=df[df["Country/Region"]==count]
								clo=list(data['Closure'])
								rec=list(data['Recovered'])
								n=len(clo)

								rec1=[]
								for i in range(30,n):
									rec1.append(100*(rec[i]/clo[i]))
								l1=[]
								for i in range(30):
									l1.append(np.nan)
								l1.extend(rec1)


								data["Recovery"]=l1
								ld.append(data)
							df_row = pd.concat(ld)
							fig = px.line(df_row,x="Date", y="Recovery",color="Country/Region")

							fig.update_layout(

									xaxis_title="Date",
									yaxis_title="Recovery",



									plot_bgcolor='white'

								)
							st.plotly_chart(fig)
							df_row.to_excel("recovery.xlsx")


							st.markdown(get_binary_file_downloader_html('recovery.xlsx', 'Recovery_Data'), unsafe_allow_html=True)




						except:
							print("Something went wrong")
					elif op=="Closure Time":
						try:
							ld=[]
							for count in multiselection:


								data=df[df["Country/Region"]==count]
								date1=[]
								date2=[]
								for i in range(30,len(data)):
									a=list(data["Closure"])[i]
									date1.append(list(data["Date"])[i])
									b=list(data["Confirmed"])[i]
									j=i
									while(b!=a and a<b):
										j=j-1
										b=list(data["Confirmed"])[j]

									date2.append(list(data["Date"])[j])
								days=[]
								for i in range(len(data)-30):
									days.append((date1[i]-date2[i]).days)
								l=[]
								for i in range(30):
									l.append(np.nan)
								l.extend(days)
								data["closure_time"]=l
								ld.append(data)

							df_row = pd.concat(ld)
							fig = px.line(df_row,x="Date", y="closure_time",color="Country/Region")

							fig.update_layout(

										xaxis_title="Date",
										yaxis_title="closure time",



										plot_bgcolor='white'

									)
							st.plotly_chart(fig)
							df_row.to_excel("Closure_Time.xlsx")


							st.markdown(get_binary_file_downloader_html('Closure_Time.xlsx', 'Closure_Time'), unsafe_allow_html=True)

						except:
							print("Something went wrong")
					elif op=="R0 - simplistic method":
						try:
							try:
								ld=[]


								for count in multiselection:



									data1=df[df["Country/Region"]==count]



									data = pd.DataFrame(data1, columns=['Date','Difference']).set_index('Date')

									data['smooth_mean(gauss, win=7)'] = data.iloc[:,0].rolling(7,
										win_type='gaussian',
										min_periods=1,
										center=True).mean(std=2).round()
									ds=list(data['smooth_mean(gauss, win=7)'])
									for i in range(len(ds)):
										if ds[i]==0:
											ds[i]=np.nan
									data['smooth_mean(gauss, win=7)']=ds
									gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
									l_7=[]
									for i in range(len(gauss)-6):
										if  m.isnan(gauss[i])==True or m.isnan(gauss[i+6])==True:
											l_7.append(np.nan)
										else:
											l_7.append(round(gauss[i+6])/round(gauss[i]))

									gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
									l_4=[]
									for i in range(len(gauss)-3):
										if  m.isnan(gauss[i])==True or m.isnan(gauss[i+3])==True :
											l_4.append(np.nan)
										else:
											l_4.append(round(gauss[i+3])/round(gauss[i]))
									N=len(data)
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
									data["Gaussian_R0_4_Days"]=gauss_4
									data["Gaussian_R0_7_Days"]=gauss_7
									col = data.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
									data['R0_Simpliste'] = col.mean(axis=1)
									R0_sim=list(data['R0_Simpliste'])
									data['Date']=data.index

									dt=list(data.Date)
									country=[count for i in range(len(dt))]
									data_per1=pd.DataFrame(list(zip(dt,R0_sim,country)),
										columns =['Date',"R0_Simpliste","Country/Region"])
									ld.append(data_per1)
							except:
								st.markdown("Sorry something went wrong. Please try to choose an other priod for  "+count)
							df_row = pd.concat(ld)
							df_row=df_row.sort_values(by=['Date'])
							ind1=min(list(df_row['Date']))

							ind2=max(list(df_row['Date']))
							n=(ind2-ind1).days
							dat=[ind1+timedelta(days=i) for i in range(n+1)]

							from_date  = st.selectbox('From :', dat )
							l2=dat[1:]
							to_date= st.selectbox('To :',l2  )
							ind1=[index for index, value in enumerate(list(df_row['Date'])) if value == from_date][0]
							ind2=[index for index, value in enumerate(list(df_row['Date'])) if value == to_date][-1]
							R0_sim=list(df_row['R0_Simpliste'])[ind1:ind2+1]
							dt=list(df_row['Date'])[ind1:ind2+1]
							coun=list(df_row['Country/Region'])[ind1:ind2+1]
							data_per=pd.DataFrame(list(zip(dt,R0_sim,coun)),
								columns =['Date',"R0","Country/Region"])

							i1=from_date
							i2=to_date
							n=(i2-i1).days
							X=[i1+timedelta(days=i) for i in range(n+1)]
							fig = px.line(data_per,x="Date", y="R0",color="Country/Region")

							fig.update_layout(
								plot_bgcolor='white',
								xaxis_title="Date",
								yaxis_title="R0 - Simplistic Method"
							)
							reference_line = go.Scatter(x=X,
													y=[1 for i in range(len(X))],
													mode="lines",
													line=go.scatter.Line(color="black"),

													showlegend=False)
							fig.add_trace(reference_line)


							fig.add_trace(reference_line)
							st.plotly_chart(fig)

							df_row=df_row.sort_values(by=['Country/Region',"Date"])
							df_row.to_excel("R0_Simplistic.xlsx")


							st.markdown(get_binary_file_downloader_html('R0_Simplistic.xlsx', 'Data'), unsafe_allow_html=True)




						except:
							print("something went wrong")
					if op=="R0 - Bettencourt & Rebeiro Method":
						try:
							try:
								ld=[]


								for count in multiselection:



									data1=df[df["Country/Region"]==count]

									data1.to_excel("Data_covid.xlsx")
									url = 'Data_covid.xlsx'
									dff = pd.read_excel(url,
														usecols=['Date', 'Country/Region', 'Difference'],
														index_col=[1,0],
														squeeze=True).sort_index()
									country_name = count
									# We create an array for every possible value of Rt
									R_T_MAX = 12
									r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

									def prepare_cases(cases, cutoff=2):
										new_cases = cases

										smoothed = new_cases.rolling(7,
											win_type='gaussian',
											min_periods=1,
											center=True).mean(std=2).round()

										idx_start = np.searchsorted(smoothed, cutoff)

										smoothed = smoothed.iloc[idx_start:]
										original = new_cases.loc[smoothed.index]

										return original, smoothed
									cases = dff.xs(country_name).rename(f"{country_name} cases")

									original, smoothed = prepare_cases(cases)
									l=[i for i in range(len(smoothed)) if smoothed[i]==0]
									for i in l:
										smoothed[i]=1
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
											if denominator!=0:

										# Execute full Bayes' Rule
												posteriors[current_day] = numerator/denominator

										# Add to the running sum of log likelihoods

												log_likelihood += np.log(denominator)

										return posteriors, log_likelihood

									# Note that we're fixing sigma to a value just for the example
									posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
									posteriors=posteriors.dropna(axis=1, how='all')
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
									coun=[count for i in range(len(values))]
									R0 = pd.DataFrame(list(zip(list(index), list(values),coun)),
												columns =['Date', 'R0',"Country/Region"])
									ld.append(R0)
							except:
								st.markdown("**Note** : Sorry, something went wrong, you can use simplistic method to visualize R0 evolution of  "+count)
							df_row = pd.concat(ld)
							df_row=df_row.sort_values(by=['Date'])
							ind1=min(list(df_row['Date']))

							ind2=max(list(df_row['Date']))
							n=(ind2-ind1).days
							dat=[ind1+timedelta(days=i) for i in range(n+1)]

							#from_date  = st.selectbox('From :', dat )
							#l2=dat[1:]
							#to_date= st.selectbox('To :',l2  )
							#ind1=[index for index, value in enumerate(list(df_row['Date'])) if value == from_date][0]
							#ind2=[index for index, value in enumerate(list(df_row['Date'])) if value == to_date][-1]
							R0_BR=list(df_row['R0'])
							dt=list(df_row['Date'])
							coun=list(df_row['Country/Region'])
							data_per=pd.DataFrame(list(zip(dt,R0_BR,coun)),
								columns =['Date',"R0","Country/Region"])

							#i1=from_date
							#i2=to_date
							#n=(i2-i1).days
							X=dat
							fig = px.line(data_per,x="Date", y="R0",color="Country/Region")

							fig.update_layout(
								plot_bgcolor='white',
								xaxis_title="Date",
								yaxis_title="R0 Bettencourt & Rebeiro"
							)
							reference_line = go.Scatter(x=X,
													y=[1 for i in range(len(X))],
													mode="lines",
													line=go.scatter.Line(color="black"),

													showlegend=False)
							fig.add_trace(reference_line)
							df_row=df_row.sort_values(by=['Country/Region',"Date"])
							st.plotly_chart(fig)
							df_row.to_excel("R0_BR.xlsx")


							st.markdown(get_binary_file_downloader_html('R0_BR.xlsx', 'Data'), unsafe_allow_html=True)
						except:
							print('something went wrong')


					if op=="Doubling Time":
						try:
							ld=[]
							for count in multiselection:

								dataa=df[df["Country/Region"]==count]
								try:
									dates = list(df["Date"])
									d=[]
									for i in dates:

										d.append(datetime.strptime(i, '%d/%m/%Y'))
									df['Date']=d
								except:
									print("Oops!")
								def irr(d1,dn,c1,cn):
									irr=(cn/c1)**(1/(dn-d1).days)-1

									return(irr)
								def DT(irr):
									p=m.log(2)/m.log(1+irr)
									return(p)
								data = pd.DataFrame(dataa, columns=['Date',"Country/Region",'Confirmed',"Closure"]).set_index('Date')
							#n1=list(data["date"]).index(list(dr['Date'])[0])
							#n2=list(data["date"]).index(list(dr['Date'])[-1])
							#st.write(n1)
							#st.write(n2)

								IRR=[]
								DoubT=[]
								l=list(data["Confirmed"])
								n=0
								for i in range(len(l)-1):
									if l[i]!=l[i+1]:
										n=i
										break

								c1=data["Confirmed"][n]
								d1=dates[n]
								for i in range(n+1,len(data)):
									cn=data["Confirmed"][i]
									dn=dates[i]
									IRR.append(irr(d1,dn,c1,cn)*100)
									DoubT.append(DT(irr(d1,dn,c1,cn)))

								DoublingTime=[]
								for i in range(n+1):
									DoublingTime.append(np.nan)
								DoublingTime.extend(DoubT)
								data1=data.copy()
								data1['Doubling Time']=DoublingTime
								data1['Date']=data1.index
								ld.append(data1)

							df_row = pd.concat(ld)
							fig = px.line(df_row,x="Date", y="Doubling Time",color="Country/Region")

							fig.update_layout(

										xaxis_title="Date",
										yaxis_title="Doubling Time",



										plot_bgcolor='white'

									)
							st.plotly_chart(fig)
							df_row.to_excel("DT.xlsx")
							st.markdown(get_binary_file_downloader_html('DT.xlsx', 'Data'), unsafe_allow_html=True)

						except:
							st.markdown("Something went wrong")
					if op=="IRR":
						try:
							ld=[]
							for count in multiselection:

								dataa=df[df["Country/Region"]==count]
								try:
									dates = list(df["Date"])
									d=[]
									for i in dates:

										d.append(datetime.strptime(i, '%d/%m/%Y'))
									df['Date']=d
								except:
									print("Oops!")
								def irr(d1,dn,c1,cn):
									irr=(cn/c1)**(1/(dn-d1).days)-1

									return(irr)

								data = pd.DataFrame(dataa, columns=['Date',"Country/Region",'Confirmed',"Closure"]).set_index('Date')
							#n1=list(data["date"]).index(list(dr['Date'])[0])
							#n2=list(data["date"]).index(list(dr['Date'])[-1])
							#st.write(n1)
							#st.write(n2)

								IRR=[]
								l=list(data["Confirmed"])
								n=0
								for i in range(len(l)-1):
									if l[i]!=l[i+1]:
										n=i
										break

								c1=data["Confirmed"][n]
								d1=dates[n]
								for i in range(n+1,len(data)):
									cn=data["Confirmed"][i]
									dn=dates[i]
									IRR.append(irr(d1,dn,c1,cn)*100)
								IRRn=[]

								for i in range(n+1):
									IRRn.append(np.nan)
								IRRn.extend(IRR)
								data['IRR']=IRRn
								data['Date']=data.index
								ld.append(data)

							df_row = pd.concat(ld)
							fig = px.line(df_row,x="Date", y="IRR",color="Country/Region")

							fig.update_layout(

										xaxis_title="Date",
										yaxis_title="IRR",



										plot_bgcolor='white'

									)
							st.plotly_chart(fig)
							df_row.to_excel("IRR.xlsx")
							st.markdown(get_binary_file_downloader_html('IRR.xlsx', 'Data'), unsafe_allow_html=True)
						except:
							st.markdown("Something went wrong")

				elif select0=="Overview":
					st.markdown("""     """)
					image = Image.open("IM.jpg")
					st.image(image,
								use_column_width=True)
					st.markdown("")
					st.subheader("""How dangerous is the virus?""")
					st.markdown("""There are three parameters to understand in order to assess the magnitude of the risk posed by this novel coronavirus:""")
					st.markdown('**•	Transmission Rate (Ro):** number of newly infected people from a single case')
					st.markdown("**•	Case Fatality Rate (CFR):**  percent of cases that result in death")
					st.markdown("•	Determine whether asymptomatic transmission is possible")
					st.markdown(""" """)
					st.subheader("How contagious is the Wuhan Coronavirus? ")
					st.markdown("The attack rate or transmissibility (how rapidly the disease spreads) of a virus is indicated by its reproductive number (Ro, pronounced R-nought or r-zero), which represents the average number of people to which a single infected person will transmit the virus.")
					st.markdown("WHO's estimated (on Jan. 23) Ro to be between 1.4 and 2.5.")
					st.markdown("Other studies have estimated a Ro between 3.6 and 4.0, and between 2.24 to 3.58")
					st.markdown("Preliminary studies had estimated Ro to be between 1.5 and 3.5.")
					st.markdown("An outbreak with a reproductive number of below 1 will gradually disappear.")
					st.markdown("For comparison, the Ro for the common flu is 1.3 and for SARS it was 2.0.")
					st.subheader("Incubation Period (how long it takes for symptoms to appear)")
					st.markdown("Symptoms of COVID-19 may appear in as few as 2 days or as long as 14 (estimated ranges vary from 2-10 days, 2-14 days, and 10-14 days), during which the virus is contagious but the patient does not display any symptom (asymptomatic transmission).")
					st.subheader("Age and conditions of Coronavirus cases")
					st.markdown("According to early estimates by China's National Health Commission (NHC), about 80% of those who died were over the age of 60 and 75% of them had pre-existing health conditions such as cardiovascular diseases and diabetes.")
					st.markdown("•	The median age of cases detected outside of China is 45 years, ranging from 2 to 74 years.")
					st.markdown("•	71% of cases were male.")
					st.markdown("A study of 138 hospitalized patients with NCIP found that the median age was 56 years (interquartile range, 42-68; range, 22-92 years) and 75 (54.3%) were men.")
					st.markdown("The WHO, in its Myth busters FAQs, addresses the question: Does the new coronavirus affect older people, or are younger people also susceptible? by answering that:")
					st.markdown("•	**People of all ages can be infected** by the novel coronavirus COVID-19.")
					st.markdown("•	**Older people**, and people with **pre-existing medical conditions** (such as asthma, diabetes, heart disease) appear to be **more vulnerable to becoming severely ill** with the virus.")







				elif select0=="By country":

					#Select country:
					from datetime import datetime


					select1= st.sidebar.selectbox('Select Country/Region:', c, key='2')

					s=select1
					st.write("#### :globe_with_meridians: Country selected: **{}**".format(s))
					"""    """
					"""    """
					#Dates
					data_cov=data_cov[data_cov['Country/Region']==select1]
					dates = list(data_cov["Date"])
					d=[]
					for i in dates:

						d.append(datetime.strptime(i, '%d/%m/%Y'))
					rad=['Confirmed cases', 'Recovered cases',"Deaths","New cases per day","Closed cases","Active cases","Lethality","Recovery","Logistic Model","R0 (simplistic Method)","R0 (Bettencourt and Rebeiro Method)","Closure Time","Doubling Time","IRR","Tests","Government Measures"]
					radio = st.sidebar.radio(label="", options=rad)



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
					dataclo=df.copy()
					# Load  data
					data = df



					#Downoad data

					df.to_excel("data.xlsx")



					if radio == 'Confirmed cases':
						st.markdown("")
						st.subheader("Confirmed cases:")
						fig = go.Figure()
						fig.add_trace(go.Bar(name="Confirmed cases", x=list(df['Date']), y=list(df['Confirmed'])))
						fig.update_layout(
							showlegend=True,

							xaxis_title="Time (Date)",
							yaxis_title="Nº of Confirmed cases",
							plot_bgcolor='white'
							)
						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)


					elif radio == "Recovered cases":
						st.markdown("")
						st.subheader("Recovered cases:")
						fig = go.Figure()
						fig.add_trace(go.Bar(name="Recovered cases", x=list(df['Date']), y=list(df['Recovered'])))
						fig.update_layout(
							showlegend=True,
							plot_bgcolor='white',
							xaxis_title="Time (Date)", yaxis_title="Nº of Recovered cases"
							)
						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "Deaths":
						st.markdown("")
						st.subheader("Deaths:")
						fig = go.Figure()
						fig.add_trace(go.Bar(name="Deaths", x=list(df['Date']), y=list(df['Deaths'])))
						fig.update_layout(
							showlegend=True,
							xaxis_title="Time (Date)",
							yaxis_title="Nº of Deaths",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "Active cases":

						st.subheader("Active cases:")
						st.markdown("")
						st.markdown("By removing deaths and recoveries from total cases, we get currently infected cases or active cases (cases still awaiting for an outcome).")
						st.markdown("**Active cases** = Confirmed Cases - Closed Cases = Confirmed Cases - Recovered cases - Deaths ")

						fig = go.Figure()
						fig.add_trace(go.Bar(name="Active cases", x=list(df['Date']), y=list(df['Actif'])))
						fig.update_layout(
							showlegend=True,
							xaxis_title="Time (Date)",
							yaxis_title="Nº of active cases",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "Closed cases":
						st.subheader("Closed cases:")
						st.markdown("")
						st.markdown("Closure of case means recovery or death.")
						st.markdown("**Closed cases** = Recovered cases + Deaths")

						fig = go.Figure()
						fig.add_trace(go.Bar(name="Closed cases", x=list(df['Date']), y=list(df['Closure'])))
						fig.update_layout(
							showlegend=True,

							xaxis_title="Time (Date)",
							yaxis_title="Nº of Closure cases",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio=="Tests":
						st.markdown("")
						st.subheader("Covid-19 testing:")
						st.markdown("Testing for COVID-19 involves inserting a 6-inch long swab (like a long Q-tip) into the cavity between the nose and mouth (nasopharyngeal swab) for 15 seconds and rotating it several times. The swabbing is then repeated on the other side of the nose to make sure enough material is collected. The swab is then inserted into a container and sent to a lab for testing.")
						image = Image.open("test.png")
						st.image(image,
							use_column_width=True)
						st.markdown("Each country does a specific number of tests every day in order to find out how the virus is spreading and try to stop the transmission of the disease. ")
						d1 = today.strftime("%Y-%m-%d")
						test=pd.read_excel("owid-covid-data.xlsx")
						#d2=list(The database of this web application is imported from['date'])[-1]
						#from datetime import datetime

						#def days_between(d1, d2):
							#d1 = datetime.strptime(d1, "%Y-%m-%d")
							#d2 = datetime.strptime(d2, "%Y-%m-%d")
							#return abs((d2 - d1).days)
						#if d1!=list(test['date'])[-1] and days_between(d1, d2)>1 :
							#url="https://covid.ourworldindata.org/data/owid-covid-data.xlsx"
							#wget.download(url, 'owid-covid-data.xlsx')
							#test=pd.read_excel("owid-covid-data.xlsx")

						test1=test[test["location"]==select1]
						st.markdown("**New tests per day :**")
						fig1=go.Figure()

						fig1.add_trace(go.Bar(name="New tests per day", x=list(test1['date']), y=list(test1['new_tests'])))
						fig1.update_layout(
							showlegend=True,

							xaxis_title="Time (Date)",
							yaxis_title="New tests",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig1)
						st.markdown("**Total tests :**")
						fig2=go.Figure()

						fig2.add_trace(go.Bar(name="Total tests", x=list(test1['date']), y=list(test1['total_tests'])))
						fig1.update_layout(
							showlegend=True,

							xaxis_title="Time (Date)",
							yaxis_title="Total tests",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig2)
						st.markdown("**Note :** If you haven't observe anything that's because tests data of this country is not availavble.")
						test1.to_excel("data_test.xlsx")
						st.markdown(get_binary_file_downloader_html('data_test.xlsx', 'Data'), unsafe_allow_html=True)

					elif radio == "New cases per day":
						st.markdown("")
						st.subheader("New cases:")
						data = pd.DataFrame(df, columns=['Date','Difference']).set_index('Date')


						data['Smoothed'] = data.iloc[:,0].rolling(7,
								win_type='gaussian',
								min_periods=1,
								center=True).mean(std=2).round()



						fig = go.Figure()
						fig.add_trace(go.Bar(name="New cases per day", x=list(data.index), y=list(data['Difference'])))
						fig.add_trace(go.Line(name="Time serie of infection (Smoothed)", x=list(data.index), y=list(data['Smoothed'])))
						data.to_excel("data.xlsx")

						fig.update_layout(
							showlegend=True,

							xaxis_title="Time (Date)",
							yaxis_title="Nº of new cases per day",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)

					#st.markdown('**Cumulative Confirmed cases:** {}  \n'.format(list(df['Confirmed'])[-1]) +
								#'**Cumulative deaths:** {}  \n'.format(list(df['Deaths'])[-1])+
								#'**Cumulative Recovered cases:** {}'.format(list(df['Recovered'])[-1]) )





					#Lethality rate


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
								yaxis_title="Lethality",



								plot_bgcolor='white'

							)
						if radio=="Lethality":
							st.subheader("☞ Evolution of Lethality rate over time in "+select1)
							"""  """
							st.markdown("Lethality is the number of deaths / over number of sick with a specific disease (x100) It is also known as **Case fatality rate**. It is the proportion of cases in a designated population of a particular disease, which die in a specified period of time.")
							st.markdown("**Lethality** = Case Fatality Rate = CFR ")
							image = Image.open("cfr.png")
							st.image(image,
							use_column_width=True)
							st.markdown("")
							st.markdown("**Note:** lethality is a better measure of clinical significance of the disease than mortality. For example: Naegleria fowleri has a much higher lethality (it will surely kill you once you get it), than heart attacks who have a higher mortality, that is, more people die from heart attacks (due to much higher prevalence of cardiac disease) rather than from N. fowleri infection (very low prevalence).")

							st.plotly_chart(fig)
							#Downoad data

							letalite.to_excel("lethality.xlsx")


							st.markdown(get_binary_file_downloader_html('lethality.xlsx', 'Lethality_Data'), unsafe_allow_html=True)
					except:
						print("Something went wrong")

					#Recovery rate:
					if radio=="Recovery":
						st.subheader("☞ Evolution of Recovery rate over time in "+select1)
						""""""
						st.markdown("""The recovery rate is determined as the ratio of the number of patients recovering with the diagnosis of COVID‐19 disease. """)
						st.markdown("**Recovery Rate** = (Recovered Cases ÷ Confirmed Cases) x 100% ")
						try:
							letal=[]
							recover1= []
							recover2=[]
							for i in range(30,n):
								recover1.append(100*(rec[i]/(clo[i])))
								recover2.append(100*(rec[i]/(conf[i])))
							l1=[]
							l2=[]
							for i in range(30):
								l1.append(np.nan)
								l2.append(np.nan)
							l1.extend(recover1)
							l2.extend(recover2)

							recover1=l1
							recover2=l2
							recovery=pd.DataFrame(list(zip(list(df['Date']),list(df['Country/Region']),recover2)),columns =['Date',"Country/Region",'Recovered/Confirmed'])
							recovery["Recovered/Closed"]=recover1
							fig = px.line(recovery,x="Date", y=['Recovered/Confirmed','Recovered/Closed'])
							fig.update_layout(

									xaxis_title="Date",
									yaxis_title="Recovery rate",
									plot_bgcolor='white'
								)

							st.plotly_chart(fig)
							#Downoad data

							recovery.to_excel("recovery.xlsx")


							st.markdown(get_binary_file_downloader_html('recovery.xlsx', 'Recovery_data'), unsafe_allow_html=True)
						except:
							print("Something went wrong")


					# Logistic Model
					elif radio=="Logistic Model":

						#try:
							df.index=[i for i in range(len(df))]

							x = df["Confirmed"].index
							y =df["Confirmed"].values

							# inflection point estimation
							inf="no"
							fl=0
							while(inf=="no"):
								try:
									dy = np.diff(y,n=2+fl) #  derivative
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

									y_logistic2 = f(x=np.arange( len(df)-np.argmax(dy))) # 60 ==> Prédiction des cas confirmés dans les futurs 2 mois.
									inf="yes"
								except:
										inf="no"
										fl=fl+1
							k=0
							while int(y_logistic2[len(y_logistic2)-1]) - int(y_logistic2[len(y_logistic2)-2]) !=0 :
								k=k+1
								y_logistic2 = f(x=np.arange( len(df)-np.argmax(dy)+k))
							confirm=list(df["Confirmed"])
							no=[np.nan for i in range(len(df),len(df)+k)]
							confirm.extend(no)
							index=[i for i in range(len(df)+k)]
							log=[]
							log.extend(y_logistic1)
							log.extend(y_logistic2)
							date=d
							cases=list(df["Difference"])
							cases.append(abs(list(df["Confirmed"])[-1]-log[len(list(df["Confirmed"]))]))
							ll=list(np.diff(log[len(list(df["Confirmed"])):]))
							cases.extend(ll)

							fin=date[-1]
							for i in range(k):
								k=fin+timedelta(days=i+1)
								date.append(k)
							log_df = pd.DataFrame(list(zip(index,date, confirm,log)),
										columns =['index', "Date",'Confirmed',"Predicted"])
							st.subheader("☞ Logistic Model ")
							st.markdown("")

							st.markdown("The models based on mathematical statistics, machine learning and deep learning have been applied to the prediction of time series of epidemic development. Logistic is often used in regression fitting of time series data due to its simple principle and efficient calculation. For example, in the Coronavirus case, Logistic growth is characterized by a slow increase in growth at the beginning, fast growth phase approaching the peak of the incidence curve, and a slow growth phase approaching the end of the outbreak (the maximum of infections).")
							st.markdown("**Logistic Function :**")
							image = Image.open("lf1.png")
							st.image(image,
							use_column_width=True)
							st.markdown("**Modeling "+ select1+" COVID-19 Cumulative Confirmed Cases:**")
							st.markdown("**An inflection point** is a point in a graph at which the concavity changes, it represents also an event that results in a significant change in the progress of a company, industry, sector, economy... and can be considered a turning point after which a dramatic change, with either positive or negative results, is expected to result.")
							st.markdown("To model the evolution of confirmed cases we use 2 logistic funtions:")
							#Logistic function: f(x) = a / (1 + e^(-b*(x-c)))
							a1=str(round(logistic_model1[0],2))
							b1=str(round(logistic_model1[1],2))
							c1=str(round(logistic_model1[2],2))
							st.markdown(" **+**  f1(x)="+a1+"/(1+e^(-"+b1+"*(x-"+c1+")))")

							a2=str(round(logistic_model2[0],2))
							b2=str(round(logistic_model2[1],2))
							c2=str(round(logistic_model2[2],2))
							st.markdown("  **+** f2(x)="+a2+"/(1+e^(-"+b2+"*(x-"+c2+")))")
							correlation_matrix = np.corrcoef(log[0:len(df)],list(log_df["Confirmed"])[0:len(df)] )
							correlation_xy = correlation_matrix[0,1]
							r_squared = correlation_xy**2
							r2="       **R²** = "+ str(round(r_squared*100,2))+"%"
							st.markdown(r2)





							fig = go.Figure()
							fig_case=go.Figure()

							fig2=go.Figure()
							reference_line = go.Scatter(x=list(log_df["Date"]),
														y= list(log_df["Confirmed"]),
														mode="lines",
														line=go.scatter.Line(color="blue"),
														name="Confimed cases",
														showlegend=True)
							fig.add_trace(reference_line)
							reference_line2 = go.Scatter(x=list(log_df["Date"])[0:len(df)],
														y= list(log_df["Confirmed"])[0:len(df)],
														mode="lines",
														line=go.scatter.Line(color="blue"),
														name="Confimed cases",
														showlegend=True)
							fig2.add_trace(reference_line2)


							reference_line = go.Scatter(x=date,
														y=log,
														mode="lines",
														line=go.scatter.Line(color="red"),
														name="Predicted",
														showlegend=True)
							fig.add_trace(reference_line)

							fig2.add_trace(go.Line(name="Logistic Function 1", x=date[0:len(y_logistic1)], y=y_logistic1))
							fig_case.add_trace(go.Bar(name="New cases per Day", x=date, y=cases))
							fig2.add_trace(go.Line(name="Logistic Function 2", x=date[len(y_logistic1):len(df)], y=y_logistic2[0:len(df)]))


							a1=list(log_df["Date"])[np.argmax(dy)]
							l1=[]
							l1.append(a1)
							a2=list(log_df["Confirmed"])[np.argmax(dy)]
							l2=[]
							l2.append(a2)

							fig2.add_trace(go.Line(name="Inflection point", x=l1,y=l2))

							fig.update_layout(
							showlegend=True,


									xaxis_title="Date",
									plot_bgcolor='white'
								)

							fig2.update_layout(
							showlegend=True,


									xaxis_title="Date",
									plot_bgcolor='white'
								)

							st.plotly_chart(fig2)


							fig_case.update_layout(
							showlegend=True,


									xaxis_title="Date",
									plot_bgcolor='white'
								)


							st.markdown("**Predict the end date of Covid-19 in "+select1+"**")
							st.markdown("The graphical representaion below shows the end date of covid-19 and the total number of confirmed cases in  "+select1)

							st.plotly_chart(fig)
							st.markdown('**The end date: **'+date[-1].strftime("%Y-%m-%d"))
							st.markdown('**The total number of confirmed cases : **'+str(int(log[-1])))
							st.markdown("")
							st.markdown("** 📊 The number of new cases per day **")
							st.markdown("We can extract and predict the number of new cases per day using logistic model:")
							st.plotly_chart(fig_case)

						#except:
							#print("Something went wrong")


					#R0 simplist method
					elif radio=="R0 (simplistic Method)":


						try:
							st.subheader("📈 Evolution of R0 using Simplist Method in "+select1)
							st.markdown("")

							st.markdown("""R0, pronounced “R naught,” is a mathematical term that indicates how contagious an infectious disease is. It’s also referred to as the reproduction number. As an infection is transmitted to new people, it reproduces itself.""")

							st.markdown("""R0 tells you the average number of people who will contract a contagious disease from one person with that disease. It specifically applies to a population of people who were previously free of infection and haven’t been vaccinated.""")

							st.markdown("""For example, if a disease has an R0 of 18, a person who has the disease will transmit it to an average of 18 other people. That replication will continue if no one has been vaccinated against the disease or is already immune to it in their community.""")
							image = Image.open("r0.png")
							st.image(image,
							use_column_width=True)
							st.markdown("***Period:***")
							data = pd.DataFrame(df, columns=['Date','Difference']).set_index('Date')

							data['smooth_mean(gauss, win=7)'] = data.iloc[:,0].rolling(7,
								win_type='gaussian',
								min_periods=1,
								center=True).mean(std=2).round()
							ds=list(data['smooth_mean(gauss, win=7)'])
							for i in range(len(ds)):
								if ds[i]==0:
									ds[i]=np.nan
							data['smooth_mean(gauss, win=7)']=ds
							gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
							l_7=[]
							for i in range(len(gauss)-6):
								if  m.isnan(gauss[i])==True or m.isnan(gauss[i+6])==True:
									l_7.append(np.nan)
								else:
									l_7.append(round(gauss[i+6])/round(gauss[i]))

							gauss=list(data['smooth_mean(gauss, win=7)'])[9:]
							l_4=[]
							for i in range(len(gauss)-3):
								if  m.isnan(gauss[i])==True or m.isnan(gauss[i+3])==True :
									l_4.append(np.nan)
								else:
									l_4.append(round(gauss[i+3])/round(gauss[i]))
							N=len(data)
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
							data["Gaussian_R0_4_Days"]=gauss_4
							data["Gaussian_R0_7_Days"]=gauss_7
							col = data.loc[: , "Gaussian_R0_4_Days":"Gaussian_R0_7_Days"]
							data['R0_Simpliste'] = col.mean(axis=1)
							R0_sim=list(data['R0_Simpliste'])
							data['Date']=data.index
							from_date  = st.selectbox('From :', list(data.Date) )
							ind1=list(data.Date).index(from_date)
							l2=list(data.Date)[int(ind1)+1:]
							to_date= st.selectbox('To :',l2  )
							ind2=list(data.Date).index(to_date)
							R0_sim=R0_sim[ind1:ind2+1]
							dt=list(data.Date)[ind1:ind2+1]
							data_per=pd.DataFrame(list(zip(dt,R0_sim)),
								columns =['Date',"R0_Simpliste"])
							fig = go.Figure()
							fig.add_trace(go.Line(name="R0 - Simplistic Method", x=list(data_per['Date']), y=list(data_per['R0_Simpliste'])))
							fig.update_layout(
							showlegend=True,

							plot_bgcolor='white',
							xaxis_title="Date",
							yaxis_title="R0 Simplistic"
						)
							reference_line = go.Scatter(x=list(data_per['Date']),
												y=[1 for i in range(len(dt))],
												mode="lines",
												line=go.scatter.Line(color="red"),

												showlegend=False)
							fig.add_trace(reference_line)
							st.plotly_chart(fig)

							from scipy.signal import argrelextrema
							data=data_per
							data['Country']=[select1 for i in range(len(data))]
							da=[data['Date'][i].strftime("%d-%m-%Y") for i in range(len(data))]
							data['Date']=da
							data.plot(x="Date",y="R0_Simpliste",label="R0", figsize=(14,5), color="m")


							peak_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.greater)
							peak_indexes = peak_indexes[0]
							plt.axhline(y=1,linestyle='--', color='black')

							# Plot peaks.
							peak_x = peak_indexes
							peak_y = np.array(list(data['R0_Simpliste']))[peak_indexes]

							# Find valleys(min).
							valley_indexes = argrelextrema(np.array(list(data['R0_Simpliste'])), np.less)
							valley_indexes = valley_indexes[0]


							# Plot valleys.
							valley_x = valley_indexes
							valley_y =  np.array(list(data['R0_Simpliste']))[valley_indexes]

							reg_x=np.union1d(valley_indexes,peak_indexes)
							reg_y=np.array(list(data['R0_Simpliste']))[reg_x]
							plt.plot(reg_x, reg_y, marker='o', linestyle='dashed', color='red', label="linear regression")
							# Save graph to file.
							plt.xlabel('Date')
							plt.legend(loc='best')

							plt.title("R0_Simpliste "+select1)
							plt.legend(loc='best')
							#path=os.path.abspath(os.getcwd())
							plt.savefig('R0_Sim.jpg')
							image = Image.open('R0_Sim.jpg')
							st.image(image, caption='R0_Simplist '+select1,
							use_column_width=True)

							#Downoad data
							st.markdown("""""")
							st.markdown("""**Note: ** World knows both a decrease in the number of new reported cases and an increase in it. In fact, some countries sometimes report zero coronavirus cases for a period of time as China, Somalia... This variance can influence the calculation of R0. That's why you can observe some missing values.    """)

							data_per.to_excel("R0_sim.xlsx")


							st.markdown(get_binary_file_downloader_html('R0_sim.xlsx', 'R0_simp Data'), unsafe_allow_html=True)
						except:
							print("Something went wrong")

					elif radio=="R0 (Bettencourt and Rebeiro Method)":

						#st.write("#### :globe_with_meridians: Country : **{}**".format(str(indi)))


						try:
							st.subheader("📈 Evolution of R0 using Bettencourt & Rebeiro Method in "+select1)
							st.markdown("")

							st.markdown("""R0, pronounced “R naught,” is a mathematical term that indicates how contagious an infectious disease is. It’s also referred to as the reproduction number. As an infection is transmitted to new people, it reproduces itself.""")

							st.markdown("""R0 tells you the average number of people who will contract a contagious disease from one person with that disease. It specifically applies to a population of people who were previously free of infection and haven’t been vaccinated.""")

							st.markdown("""For example, if a disease has an R0 of 18, a person who has the disease will transmit it to an average of 18 other people. That replication will continue if no one has been vaccinated against the disease or is already immune to it in their community.""")
							image = Image.open("r0.png")
							st.image(image,
							use_column_width=True)
							st.markdown("***R0:***")
							df = pd.read_excel("Covid.xlsx")
							df=df[df["Country/Region"]==select1]
							#Convert Date column to date type:
							l=[]
							d=list(df.Date)
							from datetime import datetime
							for i in range(len(d)):
								l.append(datetime.strptime(d[i], '%d/%m/%Y'))
							df["Date"]=l
							df.to_excel("Data_covid.xlsx")
							url = 'Data_covid.xlsx'
							df = pd.read_excel(url,
												usecols=['Date', 'Country/Region', 'Difference'],
												index_col=[1,0],
												squeeze=True).sort_index()
							country_name = select1
							# We create an array for every possible value of Rt
							R_T_MAX = 12
							r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

							def prepare_cases(cases, cutoff=2):
								new_cases = cases

								smoothed = new_cases.rolling(7,
									win_type='gaussian',
									min_periods=1,
									center=True).mean(std=2).round()

								idx_start = np.searchsorted(smoothed, cutoff)

								smoothed = smoothed.iloc[idx_start:]
								original = new_cases.loc[smoothed.index]

								return original, smoothed
							cases = df.xs(country_name).rename(f"{country_name} cases")

							original, smoothed = prepare_cases(cases)
							GAMMA = 1/7

							l=[i for i in range(len(smoothed)) if smoothed[i]==0]
							for i in l:
								smoothed[i]=1
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
									if denominator!=0:

								# Execute full Bayes' Rule
										posteriors[current_day] = numerator/denominator

								# Add to the running sum of log likelihoods

										log_likelihood += np.log(denominator)

								return posteriors, log_likelihood

							# Note that we're fixing sigma to a value just for the example
							posteriors, log_likelihood = get_posteriors(smoothed, sigma=.25)
							posteriors=posteriors.dropna(axis=1, how='all')
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
							R0.index=R0.Date
							r0=R0.copy()

							#from_date  = st.selectbox('From :', list(result.index) )
							#ind1=list(result.index).index(from_date)
							#l2=list(result.index)[ind1+1:]
							#to_date= st.selectbox('To :',l2  )
							#ind2=list(result.index).index(to_date)

							R0_BR=list(values)
							dt=list(result.index)
							coun=[select1 for i in range(len(dt))]
							data_per=pd.DataFrame(list(zip(dt,R0_BR,coun)),
								columns =['Date',"R0","Country/Region"])
							fig = go.Figure()
							fig.add_trace(go.Line(name="R0 - Bettencourt & Rebeiro ", x=list(data_per['Date']), y=list(data_per['R0'])))
							fig.update_layout(
							showlegend=True,


									xaxis_title="Date",
									yaxis_title="R0_Bettencourt & Rebeiro",
									plot_bgcolor='white'
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
							#path=os.path.abspath(os.getcwd())
							plt.savefig('R0_B&R.jpg')


							image = Image.open('R0_B&R.jpg')
							st.image(image, caption='R0_Bettencourt & Rebeiro '+select1,
									use_column_width=True)
							st.markdown("""""")
							st.markdown("""**Note: ** World knows both a decrease in the number of new reported cases and an increase in it. In fact, some countries sometimes report zero coronavirus cases for a period of time as China, Somalia... This variance can influence the calculation of R0. That's why you can observe some missing values.    """)
							#Downoad data


							st.markdown(get_binary_file_downloader_html("Data_Bettencourt_&_Ribeiro_"+select1+".xlsx", 'R0_Bettencourt& Rebeiro Data'), unsafe_allow_html=True)

						except:
							st.markdown("Sorry, something went wrong you can visualize R0 using Simplistic Method. ")
						#st.write(type(list(data_per['Date'])[0]))
					elif radio=="Closure Time":
						#Closure Time

						try:
							st.subheader("📆 Closure Time in "+select1)
							st.markdown('')
							st.markdown("**Closure Time:** represents the number of days needed to close a newly identified case (death or recovery)")
							data = pd.DataFrame(dataclo, columns=['Date','Confirmed',"Closure"]).set_index('Date')
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
							fig = go.Figure()
							fig.add_trace(go.Line(name="Closure Time ", x=list(data['Date']), y=list(data['closure_time'])))
							fig.update_layout(
							showlegend=True,



									xaxis_title="Date",
									yaxis_title="Closure Time",
									plot_bgcolor='white'
								)
							st.plotly_chart(fig)

							#Downoad data


							st.markdown(get_binary_file_downloader_html("closure_time.xlsx", 'Closure Time Data'), unsafe_allow_html=True)

						except:
							print("Something went wrong")

					#IRR:
					elif radio=="IRR" or radio=="Doubling Time":
						try:

							df = pd.read_excel("Covid.xlsx")
							df=df[df["Country/Region"]==select1]
							#dr = pd.read_excel("R0_sim.xlsx")
							#Convert Date column to date type:

							#d=list(dr["Date"])
							l=[]
							from datetime import datetime
							#for i in range(len(d)):
								#l.append(datetime.strptime(d[i], '%d-%m-%Y'))
							#dr["Date"]=l




							def irr(d1,dn,c1,cn):
								irr=(cn/c1)**(1/(dn-d1).days)-1

								return(irr)
							def DT(irr):
								p=m.log(2)/m.log(1+irr)
								return(p)
							data = pd.DataFrame(df, columns=['Date',"Country/Region",'Confirmed',"Closure"]).set_index('Date')

							x=list(df['Date'])
							dates = [datetime.strptime(i1,'%d/%m/%Y') for i1 in x]
							data["date"]=dates
							#st.write(dates[0])
							#y=list(r0['Date'])
							#dates = [datetime.strptime(i1,'%d-%m-%Y') for i1 in y]
							#st.write(dates[0])
							#r0["Date"]=dates

							#n1=list(data["date"]).index(list(dr['Date'])[0])
							#n2=list(data["date"]).index(list(dr['Date'])[-1])
							#st.write(n1)
							#st.write(n2)

							#data=data[n1:n2+1]
							IRR=[]
							DoubT=[]
							l=list(data["Confirmed"])
							n=0
							for i in range(len(l)-1):
								if l[i]!=l[i+1]:
									n=i
									break

							c1=data["Confirmed"][n]
							d1=dates[n]
							for i in range(n+1,len(data)):
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
							if radio=="Doubling Time":
								st.subheader("📆 Doubling Time in "+select1)
								st.markdown("")
								st.markdown("**The doubling time of an epidemic **  is the period of time required for the number of cases in the epidemic to double.")
								fig= go.Figure()


								reference_line = go.Scatter(x=list(data1["date"]),
															y=list(data1["Doubling Time"]),
															mode="lines",
															line=go.scatter.Line(color="blue"),
															name="Doubling Time",
															showlegend=True)
								fig.add_trace(reference_line)
								reference_line = go.Scatter(x=list(data1["date"])[n+1:],
															y=y_linear,
															mode="lines",
															line=go.scatter.Line(color="red"),
															name="Linear Regression",
															showlegend=True)
								fig.add_trace(reference_line)

								fig.update_layout(

										xaxis_title="Date",
										yaxis_title="Doubling Time",
										plot_bgcolor='white'
									)
								st.write("Linear Regression: f(Time) = "+a+"*X +"+b)
								st.write("R-squared "+r2)
								st.plotly_chart(fig)
								#Downoad data


								st.markdown(get_binary_file_downloader_html("Doubling_Time.xlsx", 'Doubling Time Data'), unsafe_allow_html=True)
							else:
								st.subheader("📉 Incidence Rate Ratio (IRR) in "+select1)
								st.markdown('')
								st.markdown('Incidence measures the proportion of the population affected by the disease at a given time, it is one of the two most used indicators in epidemiology to assess the frequency and the spead of disease.')
								fig = go.Figure()
								fig.add_trace(go.Line(name="IRR ", x=list(data['date']), y=list(data['IRR'])))
								fig.update_layout(
									showlegend=True,




										xaxis_title="Date",
										yaxis_title="IRR",
										plot_bgcolor='white'
									)

								st.plotly_chart(fig)
								#Downoad data


								st.markdown(get_binary_file_downloader_html("IRR.xlsx", 'IRR Data'), unsafe_allow_html=True)
						except:
							print("Something went wrong")
					# Create a list of possible values and multiselect menu with them in it.
					elif radio=="Government Measures":
						st.subheader("📝 Government Measures Taken in Response to COVID-19 in "+select1)
						try:
							data=pd.read_excel("measure1.xlsx")
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
							fig = go.Figure()
							fig.add_trace(go.Bar(name="New cases per day ", x=list(dd['Date']), y=list(dd['Difference']),text=list(dd['COMMENTS'])))



							#fig = px.bar(dd,x="Date", y="Difference",text='COMMENTS'

										#)
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
							fig.update_layout(
								showlegend=True,

									xaxis_title="Date",
									yaxis_title="New cases",
									plot_bgcolor='white'
								)
							st.plotly_chart(fig)
							for i in range(len(com)):
								st.markdown('**Measure ** {}  \n'.format(com[i]))
						except:
							print("Something went wrong")













			else:
				st.warning("Incorrect Username/Password")







	elif choice == "Sign Up":
		st.subheader("Create New Account")
		new_fn = st.text_input("First name")
		new_ln = st.text_input("Last name")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Sign Up"):
			create_usertable()
			add_userdata(new_fn,new_ln,new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to Log In")



if __name__ == '__main__':
	main()
