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

image = Image.open("logo.PNG")
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
				if d1!=list(dg['Date'])[-1] and days_between(d1, d2)>1 :
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

				select0= st.sidebar.selectbox('Select :', ["Overview","World Data","By country"], key='2')
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



				if select0=="Overview":
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
					"""    """
					st.write("#### :globe_with_meridians: Country selected: **{}**".format(s))
					"""    """
					"""    """
					#Dates
					data_cov=data_cov[data_cov['Country/Region']==select1]
					dates = list(data_cov["Date"])
					d=[]
					for i in dates:

						d.append(datetime.strptime(i, '%d/%m/%Y'))
					rad=['Confirmed cases', 'Recovered cases',"Deaths","New cases per day","Closed cases","Actif cases","Lethality","Recovery","Logistic Model","R0 (simplistic Method)","R0 (Bettencourt and Rebeiro Method)","Closure Time","Doubling Time","IRR","Government Measures"]
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
						fig = px.bar(df,x="Date", y="Confirmed")
						fig.update_layout(

							xaxis_title="Time (Date)",
							yaxis_title="Nº of Confirmed cases",
							plot_bgcolor='white')
						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)


					elif radio == "Recovered cases":
						st.markdown("")
						st.subheader("Recovered cases:")
						fig = px.bar(df,x="Date", y="Recovered")
						fig.update_layout(
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
						fig = px.bar(df,x="Date", y="Deaths")
						fig.update_layout(

							xaxis_title="Time (Date)",
							yaxis_title="Nº of Deaths",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "Actif cases":
						st.markdown("")
						st.subheader("Actif cases:")
						fig = px.bar(df,x="Date", y="Actif")
						fig.update_layout(

							xaxis_title="Time (Date)",
							yaxis_title="Nº of actif cases",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "Closed cases":
						st.markdown("")
						st.subheader("Closed cases:")
						fig = px.bar(df,x="Date", y="Closure")
						fig.update_layout(

							xaxis_title="Time (Date)",
							yaxis_title="Nº of Closure cases",
							plot_bgcolor='white'

						)

						st.plotly_chart(fig)
						st.subheader('Data')
						st.write(data)
						st.markdown(get_binary_file_downloader_html('data.xlsx', 'Data'), unsafe_allow_html=True)
					elif radio == "New cases per day":
						st.markdown("")
						st.subheader("New cases:")
						fig = px.bar(df,x="Date", y="Difference")
						fig.update_layout(

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
							st.markdown("**Lethality** = number of deaths / over number of sick with a specific disease (x100) It is also known as Case fatality rate. It is the proportion of cases in a designated population of a particular disease, which die in a specified period of time.")
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
						st.markdown("""**The recovery rate:** is determined as the ratio of the number of patients recovering with the diagnosis of COVID‐19 disease. """)
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
							st.markdown("")

							st.markdown("The models based on mathematical statistics, machine learning and deep learning have been applied to the prediction of time series of epidemic development. Logistic is often used in regression fitting of time series data due to its simple principle and efficient calculation. For example, in the Coronavirus case, Logistic growth is characterized by a slow increase in growth at the beginning, fast growth phase approaching the peak of the incidence curve, and a slow growth phase approaching the end of the outbreak (the maximum of infections).")

							fig = go.Figure()


							reference_line = go.Scatter(x=list(log_df["Date"]),
														y= list(log_df["Confirmed"]),
														mode="lines",
														line=go.scatter.Line(color="blue"),
														name="Confimed cases",
														showlegend=True)
							fig.add_trace(reference_line)

							reference_line = go.Scatter(x=date,
														y=log,
														mode="lines",
														line=go.scatter.Line(color="red"),
														name="Predicted",
														showlegend=True)
							fig.add_trace(reference_line)
							fig.update_layout(

									xaxis_title="Date",
									yaxis_title="Logistic",
									plot_bgcolor='white'
								)

							st.plotly_chart(fig)
						except:
							print("Something went wrong")


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
							fig = px.line(data_per,x="Date", y="R0_Simpliste")

							fig.update_layout(
							plot_bgcolor='white',
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
							path=os.path.abspath(os.getcwd())
							plt.savefig(path+'\\R0_Sim.jpg')
							image = Image.open(path+'\\R0_Sim.jpg')
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
							st.markdown("***Period:***")
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

							def prepare_cases(cases, cutoff=10):
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

							from_date  = st.selectbox('From :', list(result.index) )
							ind1=list(result.index).index(from_date)
							l2=list(result.index)[ind1+1:]
							to_date= st.selectbox('To :',l2  )
							ind2=list(result.index).index(to_date)
							coun=[select1 for i in range(ind1,ind2+1)]
							R0_BR=list(values)[ind1:ind2+1]
							dt=list(result.index)[ind1:ind2+1]
							data_per=pd.DataFrame(list(zip(dt,R0_BR,coun)),
								columns =['Date',"R0","Country/Region"])
							fig = px.line(data_per,x="Date", y="R0")

							fig.update_layout(

									xaxis_title="Date",
									yaxis_title="R0_Bettencourt&Rebeiro",
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
							path=os.path.abspath(os.getcwd())
							plt.savefig(path+'\\R0_B&R.jpg')


							image = Image.open(path+'\\R0_B&R.jpg')
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
							fig = px.bar(data,x="Date", y="closure_time")

							fig.update_layout(

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
							dr = pd.read_excel("R0_sim.xlsx")
							#Convert Date column to date type:

							d=list(dr["Date"])
							l=[]
							from datetime import datetime
							for i in range(len(d)):
								l.append(datetime.strptime(d[i], '%d-%m-%Y'))
							dr["Date"]=l




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

							n1=list(data["date"]).index(list(dr['Date'])[0])
							n2=list(data["date"]).index(list(dr['Date'])[-1])
							#st.write(n1)
							#st.write(n2)

							data=data[n1:n2+1]
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
								fig = px.line(data,x="date", y='IRR')


								fig.update_layout(

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
							fig.update_layout(

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
