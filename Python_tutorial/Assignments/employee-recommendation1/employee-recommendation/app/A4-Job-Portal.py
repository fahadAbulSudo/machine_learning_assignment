import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
#import numpy as np
import pickle as pkl
import pdfkit
import base64
import os
import json
import uuid
import glob
import re
#from IPython.display import HTML
#import os.path
from scoreFunctionList import  recommendTopMatch, similarCandidates
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from streamlit_option_menu import option_menu

# def recurringInputKeys(nameKey, n_max=20):
#   done_button = False
#   inputList = []
#   id = 0
#   while not done_button:
#     if id >= n_max:
#       break
#     if id == 1:
#       st.write("You have selected the following " + nameKey + ": ")
#     if id >= 1:
#       st.write("- " + submission)
#     uniquekey = nameKey + str(id)
#     submission = st.text_area("Enter " + nameKey + " one at a time: ", key=nameKey)
#     add_button = st.form_submit_button(label='Add Skill')
#     if add_button:
#       inputList.append(submission)
#     done_button = st.form_submit_button(label='Done!')
#     id = id + 1
#     del(st.session_state[nameKey])

#   if id >= n_max:
#     st.write("You have reached max number of skills that can be added for this section..")

#   return inputList

# user input cleaning

# FIX ME: Take these inputs from streamlit. These are dummy inputs.
# aws,   python ,,  web  developer  ...
# java", "frontend development"

def download_button_custom(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pkl.dumps(object_to_download)
        except pkl.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

def text_pre_processing(text):
  clean = text.split(",")
  clean = [x.strip() for x in clean]
  return clean

# user favorite id processing
def ID_pre_processing(text):
  clean = text.split(",")
  lst = [int(x) for x in clean]
  return lst

# make the formatted dataframe
def visualDFMaker(DF, similar=False):
  #print(recommendDF[scoreHeads[0]])
  # # some example urls
  # pd.set_option('display.max_colwidth', -1)

  # ONLY FOR EXAMPLE
  # url_ex = ['http://www.google.com', 'http://www.youtube.com', 'http://www.github.com'
  # , 'http://www.kaggle.com', 'http://www.instagram.com']
  # DF['URLs'] = DF["Name"]
  # DF['URLs'].iloc[0:5] = url_ex
  DF['File'] = DF.apply(lambda x: make_clickable(x['URLs'], x['File']), axis=1)
  #print(DF["URLs"])
  if not similar:
    passList = DF.index[DF[scoreHeads[0]] >= THRESH_RECOMMENDATION]
    showDF = DF.loc[passList]#show 10 rows
  else:
    showDF = DF

  yellowList = showDF.index[showDF[scoreHeads[1]] >= YELLOW_RECOMMEND_MUST]
  greenList = showDF.index[(showDF[scoreHeads[1]] >= GREEN_RECOMMEND_MUST)
  & (showDF[scoreHeads[3]] >= GREEN_RECOMMEND_EXP)]
  yellowList = [x for x in yellowList if x not in greenList]
  showDFShort = showDF[shortHeads]
  showExcel = showDFShort.copy()
  showExcel['File'] = showDF["URLs"]
  showExcel.rename(columns={"File":"File Link"}, inplace=True)
  #showDFShort.rename(columns={"Name":"Name & Profile"}, inplace=True)
  # anonymous function
  def f(x):
    return highlight_rows(x, yellowList, greenList)
  #showDF = showDF.style.format({'File': make_clickable}, precision=1).apply(f, axis=1)
  showDF = showDF.style.format(precision=1).apply(f, axis=1)
  showDFShort = showDFShort.style.format(precision=1).apply(f, axis=1)
  showExcel = showExcel.style.format(precision=1).apply(f, axis=1)

  return showDFShort, showExcel, showDF

def highlight_rows(row, yellowList, greenList):
  idx = row.name
  # print(idx)
  if idx in greenList:
    color = '#BAFFC9' # green
  elif idx in yellowList:
    color = '#FFF296' # light yellow
  else:
    color = '#EFEFEF' # very light grey
  return ['background-color: {}'.format(color) for r in row]

def make_clickable(url, name):
    if name == 'AFour Profile - Sudhir Padalkar.docx' or name == 'AFour Profile - Roshan Wakode.odt':
      return '<a href={}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,name)
    else:
      return '<a href="file:///{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,name)

# def make_clickable(val):
#     # target _blank to open new window
#     return '<a target="_blank" href="{}">{}</a>'.format(val, val)

@st.cache
def convertNSave(df, dfexcel, similar=False):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if similar == True:
      dfexcel.to_excel(PATH_EXCEL_SIM_OP)
      df.to_html(PATH_HTML_SIM_OP)
      #pdfkit.from_file(PATH_HTML_SIM_OP, PATH_PDF_SIM_OP) 
    else:
      dfexcel.to_excel(PATH_EXCEL_OP)
      df.to_html(PATH_HTML_OP)
      #pdfkit.from_file(PATH_HTML_OP, PATH_PDF_OP)
    
if __name__ == "__main__":
  # to get the location of the current python file
  PATH_BASE = os.path.dirname(os.path.abspath(__file__))
  # load candidate docs from monthly pickle file
  PATH_PROCESSDF = os.path.join(PATH_BASE,'../output/processDF.pkl')

  # TO LOAD CACHED/DOWNLOAD FILES
  PATH_TEMP = os.path.join(PATH_BASE,'temp/tempRec.pkl')
  PATH_EXCEL_OP = os.path.join(PATH_BASE,'temp/tempExcel.xlsx')
  PATH_HTML_OP = os.path.join(PATH_BASE,'temp/tempHTML.html')
  #PATH_PDF_OP = os.path.join(PATH_BASE,'temp/tempPDF.pdf')
  PATH_EXCEL_SIM_OP = os.path.join(PATH_BASE,'temp/tempSimExcel.xlsx')
  PATH_HTML_SIM_OP = os.path.join(PATH_BASE,'temp/tempSimHTML.html')
  #PATH_PDF_SIM_OP = os.path.join(PATH_BASE,'temp/tempSimPDF.pdf')

  # CONFIGS
  THRESH_RECOMMENDATION = 70. #0
  GREEN_RECOMMEND_MUST = 100. #1
  GREEN_RECOMMEND_EXP = 70. #2
  YELLOW_RECOMMEND_MUST = 70. #3
  shortHeads = ["Name", "Role", "Experience", "File"] #4
  scoreHeads = ["Compatibility %", "Must Have Skills %", "Good to Have %", "Experience Match %"] #5
  IDENTIFIER = "original id" #6

  moreHeads = shortHeads.copy()
  moreHeads.append(IDENTIFIER)

  # load profiles
  with open(PATH_PROCESSDF, "rb") as file:
    processDF = pkl.load(file)
  # delete previous session if any
  if st.session_state == {}:
    files = glob.glob(PATH_BASE+'/temp/*')
    #print(files)
    for f in files:
        os.remove(f)

  with st.sidebar:
      selected = option_menu(
          menu_title="Main Menu",  # required
          options=["Recommend Profiles", "Similar Profiles"],  # required
          #icons=["house", "book", "envelope"],  # optional
          menu_icon="cast",  # optional
          default_index=0,  # optional
      )

  if selected == "Recommend Profiles":
      st.subheader('Welcome To')
      st.title("The A4 Employee Recommendation Application")
      with st.form(key='recommend', clear_on_submit=False):
        must_skill = st.text_area("Enter comma (,) separated must have skills: ")
        Good2have_skill = st.text_area("Enter comma (,) separated good to have skills: ")
        minEx = st.number_input(label='Enter Minimum Experience required in years: ',step=1.,format="%.2f")
        submit_button = st.form_submit_button(label='Show Recommendations!')
    
      noInputFlag = False
      reload = False
      if os.path.isfile(PATH_TEMP) and not submit_button:
        # check if there are previous inputs
        prevSession = pkl.load(open(PATH_TEMP, 'rb'))
        if type(prevSession) == dict:
          userMust = prevSession['must']
          userGood = prevSession['good']
          minEx = prevSession['exp']
          reload = True
  
      if submit_button:
        userMust = []
        userGood = []

        userMust = text_pre_processing(must_skill)
        userGood = text_pre_processing(Good2have_skill)
        # defaults and warnings
        if userMust == ['']:
          userMust = ['WARNING: NO INPUT PROVIDED. PROVIDE AT LEAST 1 MUST HAVE SKILL TO PROCEED..']
          noInputFlag = True
        if userGood == ['']:
          userGood = ['']
      
      if reload or submit_button:
        st.subheader("You have selected the following criteria: ")
        st.write("Must have skills:")
        for i in userMust:
          st.markdown("- " + i)
        # Wait for mandatory inputs
        if noInputFlag:
          st.stop()
        st.write("Good to have skills:")
        for i in userGood:
          st.markdown("- " + i)
        st.write("Minimum Experience:  ", minEx, " year(s)")
        recommendDF = recommendTopMatch(userMust, userGood, minEx, processDF)

        #showDF = recommendDF[visualHeads].style.format(precision=1)
        showDF, excelDF, _ = visualDFMaker(recommendDF)

        # DOWNLOAD REPORTS
        convertNSave(showDF, excelDF)
        
        st.subheader("Export Reports: ")
        with open(PATH_EXCEL_OP, 'rb') as f:
          st.download_button('Download Excel Report 📎', f, file_name='A4recommended.xlsx')
        #with open(PATH_PDF_OP, 'rb') as f:
          #st.download_button('Download PDF Report 📎', f, file_name='A4recommended.pdf')
        with open(PATH_HTML_OP, 'rb') as f:
          st.download_button('Download HTML Report 📎', f, file_name='A4recommended.html')
        
        #print(showDF)
        #st.dataframe(showDF)
        st.write(showDF.to_html(escape=False, index=False), unsafe_allow_html=True)
        saveOptions = {'must': userMust, 'good': userGood, 'exp': minEx, 'recommend': recommendDF}
        st.session_state = saveOptions
        if not reload:
          pkl.dump(saveOptions, open(PATH_TEMP, 'wb'))

  #st.write(st.session_state)
  if selected == "Similar Profiles":
    if os.path.isfile(PATH_TEMP):
      st.title(f"Find Similar Employees")
      with st.form(key='similar', clear_on_submit=False):
        with open(PATH_TEMP, 'rb') as f:
          recommendDF = pkl.load(f)['recommend']
        
        _, _, showDF = visualDFMaker(recommendDF)
        showDF = showDF.data[moreHeads]
        pd.pandas.set_option("display.precision", 1)
        gd = GridOptionsBuilder.from_dataframe(showDF[moreHeads[0:-2]])
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()
        grid_table = AgGrid(showDF, height=450, gridOptions=gridoptions,
        fit_columns_on_grid_load=False, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        update_mode=GridUpdateMode.SELECTION_CHANGED)

        #st.write('## Selected')
        selected_row = grid_table["selected_rows"]
        del_key = '_selectedRowNodeInfo'
        for items in selected_row:
          if del_key in items:
              del items[del_key]
        #selected_row.drop(columns=["_selectedRowNodeInfo"], axis=1, inplace=True)
        ID = [ sub['original id'] for sub in selected_row ]
        #st.dataframe(selected_row)
        #print(selected_row)
        #print(ID)
        #st.dataframe(recommendDF) 
        #id = st.text_area("Enter comma ids from recommended dataframe:")
        submit_button1 = st.form_submit_button(label='Show Similar Profiles')
       
      if submit_button1:
        # Check mandatory inputs
        if ID == []:
          st.write("WARNING: SELECT AT LEAST 1 CANDIDATE PROFILE TO PROCEED..")
          st.stop()
        st.subheader("Based on your selection, Following candidates might interest you:")
        #for i in ID:
          #st.markdown("- " + str(i))
        similarDF = similarCandidates(processDF, recommendDF, ID)
        showDF, excelDF, _ = visualDFMaker(similarDF, similar=True)
        st.write(showDF.to_html(escape=False, index=False), unsafe_allow_html=True)

        convertNSave(showDF, excelDF, similar=True)
        st.subheader("Export Suggestions: ")

        # with open(PATH_EXCEL_SIM_OP, 'rb') as f:
        #   st.download_button('Download Excel Report', f, file_name='A4similar.xlsx')
        # with open(PATH_HTML_SIM_OP, 'rb') as f:
        #   st.download_button('Download HTML Report', f, file_name='A4similar.html')
        
        with open(PATH_EXCEL_SIM_OP, 'rb') as f:
          s = f.read()
          download_button_str = download_button_custom(s, 'A4similar.xlsx', 'Download Excel Report 📎')
          st.markdown(download_button_str, unsafe_allow_html=True)
        #with open(PATH_PDF_SIM_OP, 'rb') as f:
          #s = f.read()
          #download_button_str = download_button_custom(s, 'A4similar.pdf', 'Download PDF Report 📎')
          #st.markdown(download_button_str, unsafe_allow_html=True)
        with open(PATH_HTML_SIM_OP, 'rb') as f:
          s = f.read()
          download_button_str = download_button_custom(s, 'A4similar.html', 'Download HTML Report 📎')
          st.markdown(download_button_str, unsafe_allow_html=True)
  
    else:
      st.write("First Generate Recommendations in 'Recommend Profiles' Tab to See Other Similar Profiles..")
