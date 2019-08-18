from data_utils.vocab import Vocabulary

NERLabelMapper = Vocabulary(True)
NERLabelMapper.add('O')
NERLabelMapper.add('B-MISC')
NERLabelMapper.add('I-MISC')
NERLabelMapper.add('B-PERSON')
NERLabelMapper.add('I-PERSON')
NERLabelMapper.add('B-ORGANIZATION')
NERLabelMapper.add('I-ORGANIZATION')
NERLabelMapper.add('B-LOCATION')
NERLabelMapper.add('I-LOCATION')
NERLabelMapper.add('B-ARTIFACT')
NERLabelMapper.add('I-ARTIFACT')
NERLabelMapper.add('B-DATE')
NERLabelMapper.add('I-DATE')
NERLabelMapper.add('B-TIME')
NERLabelMapper.add('I-TIME')
NERLabelMapper.add('B-MONEY')
NERLabelMapper.add('I-MONEY')
NERLabelMapper.add('B-PERCENT')
NERLabelMapper.add('I-PERCENT')
NERLabelMapper.add('X')
NERLabelMapper.add('[CLS]')
NERLabelMapper.add('[SEP]')

NERALLLabelMapper = Vocabulary(True)
NERALLLabelMapper.add('O')
NERALLLabelMapper.add('X')
NERALLLabelMapper.add('[CLS]')
NERALLLabelMapper.add('[SEP]')
NERALLLabelMapper.add('B-Airport')
NERALLLabelMapper.add('B-Constellation')
NERALLLabelMapper.add('B-Nationality')
NERALLLabelMapper.add('I-Magazine')
NERALLLabelMapper.add('I-Sea')
NERALLLabelMapper.add('B-Seismic_Intensity')
NERALLLabelMapper.add('B-Mineral')
NERALLLabelMapper.add('B-Money_Form')
NERALLLabelMapper.add('B-Phone_Number')
NERALLLabelMapper.add('B-Timex_Other')
NERALLLabelMapper.add('B-Theory')
NERALLLabelMapper.add('I-Style')
NERALLLabelMapper.add('I-Sports_League')
NERALLLabelMapper.add('I-Name_Other')
NERALLLabelMapper.add('B-Nature_Color')
NERALLLabelMapper.add('I-Era')
NERALLLabelMapper.add('I-Newspaper')
NERALLLabelMapper.add('I-N_Animal')
NERALLLabelMapper.add('I-Printing_Other')
NERALLLabelMapper.add('I-Natural_Object_Other')
NERALLLabelMapper.add('B-Natural_Disaster')
NERALLLabelMapper.add('B-Car_Stop')
NERALLLabelMapper.add('B-Country')
NERALLLabelMapper.add('B-Port')
NERALLLabelMapper.add('I-Compound')
NERALLLabelMapper.add('B-Mammal')
NERALLLabelMapper.add('I-Fish')
NERALLLabelMapper.add('I-Person')
NERALLLabelMapper.add('I-Mountain')
NERALLLabelMapper.add('B-Volume')
NERALLLabelMapper.add('B-Park')
NERALLLabelMapper.add('I-Clothing')
NERALLLabelMapper.add('I-Measurement_Other')
NERALLLabelMapper.add('B-Title_Other')
NERALLLabelMapper.add('B-Measurement_Other')
NERALLLabelMapper.add('I-Period_time')
NERALLLabelMapper.add('I-Nature_Color')
NERALLLabelMapper.add('B-N_Animal')
NERALLLabelMapper.add('B-N_Location_Other')
NERALLLabelMapper.add('B-God')
NERALLLabelMapper.add('B-Research_Institute')
NERALLLabelMapper.add('B-Animal_Part')
NERALLLabelMapper.add('I-Railroad')
NERALLLabelMapper.add('I-Bridge')
NERALLLabelMapper.add('B-Age')
NERALLLabelMapper.add('I-Fungus')
NERALLLabelMapper.add('I-Natural_Phenomenon_Other')
NERALLLabelMapper.add('B-Planet')
NERALLLabelMapper.add('B-Class')
NERALLLabelMapper.add('B-Ethnic_Group_Other')
NERALLLabelMapper.add('I-Game')
NERALLLabelMapper.add('I-Planet')
NERALLLabelMapper.add('I-Multiplication')
NERALLLabelMapper.add('B-URL')
NERALLLabelMapper.add('I-Rule_Other')
NERALLLabelMapper.add('I-Aircraft')
NERALLLabelMapper.add('B-Movement')
NERALLLabelMapper.add('B-Family')
NERALLLabelMapper.add('B-Position_Vocation')
NERALLLabelMapper.add('I-Conference')
NERALLLabelMapper.add('I-Religious_Festival')
NERALLLabelMapper.add('I-Time_Top_Other')
NERALLLabelMapper.add('B-Speed')
NERALLLabelMapper.add('I-Lake')
NERALLLabelMapper.add('I-Weapon')
NERALLLabelMapper.add('B-Doctrine_Method_Other')
NERALLLabelMapper.add('I-Physical_Extent')
NERALLLabelMapper.add('I-Company')
NERALLLabelMapper.add('I-Island')
NERALLLabelMapper.add('I-Ordinal_Number')
NERALLLabelMapper.add('I-Academic')
NERALLLabelMapper.add('B-Domestic_Region')
NERALLLabelMapper.add('I-N_Event')
NERALLLabelMapper.add('B-Law')
NERALLLabelMapper.add('I-Material')
NERALLLabelMapper.add('I-Movement')
NERALLLabelMapper.add('B-Newspaper')
NERALLLabelMapper.add('I-Animal_Part')
NERALLLabelMapper.add('B-Language_Other')
NERALLLabelMapper.add('I-Road')
NERALLLabelMapper.add('B-Material')
NERALLLabelMapper.add('B-Music')
NERALLLabelMapper.add('I-Zoo')
NERALLLabelMapper.add('I-Living_Thing_Other')
NERALLLabelMapper.add('I-Tumulus')
NERALLLabelMapper.add('B-Amusement_Park')
NERALLLabelMapper.add('B-Drug')
NERALLLabelMapper.add('B-Postal_Address')
NERALLLabelMapper.add('I-Treaty')
NERALLLabelMapper.add('I-Cabinet')
NERALLLabelMapper.add('B-Style')
NERALLLabelMapper.add('I-Timex_Other')
NERALLLabelMapper.add('B-Sports_League')
NERALLLabelMapper.add('I-Market')
NERALLLabelMapper.add('B-N_Flora')
NERALLLabelMapper.add('I-Ship')
NERALLLabelMapper.add('B-Clothing')
NERALLLabelMapper.add('B-Natural_Phenomenon_Other')
NERALLLabelMapper.add('B-Element')
NERALLLabelMapper.add('I-Age')
NERALLLabelMapper.add('B-Frequency')
NERALLLabelMapper.add('B-Numex_Other')
NERALLLabelMapper.add('B-N_Product')
NERALLLabelMapper.add('B-Aircraft')
NERALLLabelMapper.add('I-Reptile')
NERALLLabelMapper.add('I-Theater')
NERALLLabelMapper.add('I-Countx_Other')
NERALLLabelMapper.add('I-Money_Form')
NERALLLabelMapper.add('B-GPE_Other')
NERALLLabelMapper.add('B-N_Organization')
NERALLLabelMapper.add('B-River')
NERALLLabelMapper.add('I-County')
NERALLLabelMapper.add('B-Company')
NERALLLabelMapper.add('I-Country')
NERALLLabelMapper.add('B-Bridge')
NERALLLabelMapper.add('B-Dish')
NERALLLabelMapper.add('B-Multiplication')
NERALLLabelMapper.add('I-Disease_Other')
NERALLLabelMapper.add('B-Sports_Facility')
NERALLLabelMapper.add('I-Flora')
NERALLLabelMapper.add('B-Position_vocation')
NERALLLabelMapper.add('B-Ordinal_Number')
NERALLLabelMapper.add('I-Archaeological_Place_Other')
NERALLLabelMapper.add('B-Animal_Disease')
NERALLLabelMapper.add('I-Space')
NERALLLabelMapper.add('I-Doctrine_Method_Other')
NERALLLabelMapper.add('B-Offense')
NERALLLabelMapper.add('I-Park')
NERALLLabelMapper.add('B-Company_Group')
NERALLLabelMapper.add('B-Freguency')
NERALLLabelMapper.add('B-Award')
NERALLLabelMapper.add('I-Spa')
NERALLLabelMapper.add('I-Position_Vocation')
NERALLLabelMapper.add('I-Facility_Part')
NERALLLabelMapper.add('B-Product_Other')
NERALLLabelMapper.add('I-Line_Other')
NERALLLabelMapper.add('I-Nationality')
NERALLLabelMapper.add('I-School_Age')
NERALLLabelMapper.add('B-Line_Other')
NERALLLabelMapper.add('B-GOE_Other')
NERALLLabelMapper.add('B-Book')
NERALLLabelMapper.add('B-Island')
NERALLLabelMapper.add('B-Sport')
NERALLLabelMapper.add('I-Day_Of_Week')
NERALLLabelMapper.add('I-Phone_Number')
NERALLLabelMapper.add('I-Car_Stop')
NERALLLabelMapper.add('B-Era')
NERALLLabelMapper.add('I-N_Country')
NERALLLabelMapper.add('I-URL')
NERALLLabelMapper.add('I-Offence')
NERALLLabelMapper.add('B-Bay')
NERALLLabelMapper.add('I-Vehicle_Other')
NERALLLabelMapper.add('I-Email')
NERALLLabelMapper.add('I-ID_Number')
NERALLLabelMapper.add('B-Name_Other')
NERALLLabelMapper.add('B-Museum')
NERALLLabelMapper.add('B-Religious_Festival')
NERALLLabelMapper.add('B-Compound')
NERALLLabelMapper.add('I-Calorie')
NERALLLabelMapper.add('I-Element')
NERALLLabelMapper.add('I-Position_vocation')
NERALLLabelMapper.add('I-Location_Other')
NERALLLabelMapper.add('I-Event_Other')
NERALLLabelMapper.add('I-Period_Day')
NERALLLabelMapper.add('I-Stock_Index')
NERALLLabelMapper.add('I-GOE_Other')
NERALLLabelMapper.add('I-Numex_Other')
NERALLLabelMapper.add('I-Province')
NERALLLabelMapper.add('I-Color_Other')
NERALLLabelMapper.add('B-Unit_Other')
NERALLLabelMapper.add('B-Physical_Extent')
NERALLLabelMapper.add('B-Mollusc_Arthropod')
NERALLLabelMapper.add('B-Treaty')
NERALLLabelMapper.add('I-Museum')
NERALLLabelMapper.add('I-Postal_Address')
NERALLLabelMapper.add('I-Periodx_Other')
NERALLLabelMapper.add('I-Pro_Sports_Organization')
NERALLLabelMapper.add('B-Geological_Region_Other')
NERALLLabelMapper.add('I-Time')
NERALLLabelMapper.add('I-Book')
NERALLLabelMapper.add('I-Period_Time')
NERALLLabelMapper.add('B-Period_Time')
NERALLLabelMapper.add('I-Intensity')
NERALLLabelMapper.add('B-Zoo')
NERALLLabelMapper.add('B-Location_Other')
NERALLLabelMapper.add('I-Culture')
NERALLLabelMapper.add('I-Government')
NERALLLabelMapper.add('B-Period_Month')
NERALLLabelMapper.add('B-Natural_Object_Other')
NERALLLabelMapper.add('I-Mollusc_Arthropod')
NERALLLabelMapper.add('I-Organization_Other')
NERALLLabelMapper.add('I-Train')
NERALLLabelMapper.add('B-Continental_Region')
NERALLLabelMapper.add('I-Facility_Other')
NERALLLabelMapper.add('I-Movie')
NERALLLabelMapper.add('B-Printing_Other')
NERALLLabelMapper.add('B-N_Country')
NERALLLabelMapper.add('I-N_Flora')
NERALLLabelMapper.add('I-Freguency')
NERALLLabelMapper.add('I-N_Organization')
NERALLLabelMapper.add('B-Conference')
NERALLLabelMapper.add('B-Email')
NERALLLabelMapper.add('I-Amusement_Park')
NERALLLabelMapper.add('I-Broadcast_Program')
NERALLLabelMapper.add('I-Product_Other')
NERALLLabelMapper.add('B-Character')
NERALLLabelMapper.add('I-Ethnic_Group_Other')
NERALLLabelMapper.add('I-Dish')
NERALLLabelMapper.add('B-Market')
NERALLLabelMapper.add('B-Calorie')
NERALLLabelMapper.add('I-Award')
NERALLLabelMapper.add('I-Occasion_Other')
NERALLLabelMapper.add('I-N_Natural_Object_Other')
NERALLLabelMapper.add('I-Period_Year')
NERALLLabelMapper.add('B-Movie')
NERALLLabelMapper.add('B-Period_Week')
NERALLLabelMapper.add('B-Period_Day')
NERALLLabelMapper.add('I-Region_Other')
NERALLLabelMapper.add('B-Time')
NERALLLabelMapper.add('B-Point')
NERALLLabelMapper.add('B-Tunnel')
NERALLLabelMapper.add('B-Event_Other')
NERALLLabelMapper.add('I-Research_Institute')
NERALLLabelMapper.add('B-Station')
NERALLLabelMapper.add('I-Sports_Facility')
NERALLLabelMapper.add('B-Reptile')
NERALLLabelMapper.add('I-Art_Other')
NERALLLabelMapper.add('I-Drug')
NERALLLabelMapper.add('I-Airport')
NERALLLabelMapper.add('I-School')
NERALLLabelMapper.add('I-Water_Route')
NERALLLabelMapper.add('I-Law')
NERALLLabelMapper.add('B-Living_Thing_Part_Other')
NERALLLabelMapper.add('B-Political_Organization_Other')
NERALLLabelMapper.add('I-Point')
NERALLLabelMapper.add('I-Domestic_Region')
NERALLLabelMapper.add('B-Corporation_Other')
NERALLLabelMapper.add('B-Spa')
NERALLLabelMapper.add('B-N_Natural_Object_Other')
NERALLLabelMapper.add('B-Date')
NERALLLabelMapper.add('B-Period_time')
NERALLLabelMapper.add('B-Food_Other')
NERALLLabelMapper.add('I-Bay')
NERALLLabelMapper.add('B-Ship')
NERALLLabelMapper.add('I-Offense')
NERALLLabelMapper.add('I-War')
NERALLLabelMapper.add('B-N_Event')
NERALLLabelMapper.add('B-Sea')
NERALLLabelMapper.add('I-Family')
NERALLLabelMapper.add('B-Water_Route')
NERALLLabelMapper.add('B-Archaeological_Place_Other')
NERALLLabelMapper.add('B-Spaceship')
NERALLLabelMapper.add('B-Temperature')
NERALLLabelMapper.add('I-City')
NERALLLabelMapper.add('B-School_Age')
NERALLLabelMapper.add('B-Disease_Other')
NERALLLabelMapper.add('B-Fish')
NERALLLabelMapper.add('I-Food')
NERALLLabelMapper.add('B-Space')
NERALLLabelMapper.add('I-Frequency')
NERALLLabelMapper.add('B-Latitude_Longtitude')
NERALLLabelMapper.add('I-Public_Institution')
NERALLLabelMapper.add('I-Show_Organization')
NERALLLabelMapper.add('I-River')
NERALLLabelMapper.add('I-Period_Month')
NERALLLabelMapper.add('I-Station')
NERALLLabelMapper.add('B-Facility_Part')
NERALLLabelMapper.add('B-Countx_Other')
NERALLLabelMapper.add('I-National_Language')
NERALLLabelMapper.add('I-Seismic_Intensity')
NERALLLabelMapper.add('B-Currency')
NERALLLabelMapper.add('B-Vehicle_Other')
NERALLLabelMapper.add('B-Lake')
NERALLLabelMapper.add('B-Game')
NERALLLabelMapper.add('I-Money')
NERALLLabelMapper.add('I-Geological_Region_Other')
NERALLLabelMapper.add('B-War')
NERALLLabelMapper.add('B-N_Facility')
NERALLLabelMapper.add('I-Rank')
NERALLLabelMapper.add('B-Road')
NERALLLabelMapper.add('I-Sports_Organization_Other')
NERALLLabelMapper.add('B-Weapon')
NERALLLabelMapper.add('B-Plan')
NERALLLabelMapper.add('B-Show')
NERALLLabelMapper.add('I-Astral_Body_Other')
NERALLLabelMapper.add('B-Cabinet')
NERALLLabelMapper.add('I-Language_Other')
NERALLLabelMapper.add('I-International_Organization')
NERALLLabelMapper.add('B-Occasion_Other')
NERALLLabelMapper.add('I-Volume')
NERALLLabelMapper.add('I-Sport')
NERALLLabelMapper.add('B-International_Organization')
NERALLLabelMapper.add('B-Person')
NERALLLabelMapper.add('B-School')
NERALLLabelMapper.add('I-Period_Week')
NERALLLabelMapper.add('I-Show')
NERALLLabelMapper.add('I-Plan')
NERALLLabelMapper.add('B-Flora_Part')
NERALLLabelMapper.add('B-Food')
NERALLLabelMapper.add('B-Fungus')
NERALLLabelMapper.add('B-Periodx_Other')
NERALLLabelMapper.add('B-Theater')
NERALLLabelMapper.add('B-Public_Institution')
NERALLLabelMapper.add('I-Decoration')
NERALLLabelMapper.add('I-Character')
NERALLLabelMapper.add('I-Constellation')
NERALLLabelMapper.add('B-Rule_Other')
NERALLLabelMapper.add('B-City')
NERALLLabelMapper.add('I-Mineral')
NERALLLabelMapper.add('I-Theory')
NERALLLabelMapper.add('I-Political_Organization_Other')
NERALLLabelMapper.add('B-Sports_Organization_Other')
NERALLLabelMapper.add('I-Music')
NERALLLabelMapper.add('B-Insect')
NERALLLabelMapper.add('I-Bird')
NERALLLabelMapper.add('B-Government')
NERALLLabelMapper.add('B-Canal')
NERALLLabelMapper.add('B-Time_Top_Other')
NERALLLabelMapper.add('I-Food_Other')
NERALLLabelMapper.add('I-Speed')
NERALLLabelMapper.add('B-Astral_Body_Other')
NERALLLabelMapper.add('B-Stock_Index')
NERALLLabelMapper.add('B-Percent')
NERALLLabelMapper.add('B-Star')
NERALLLabelMapper.add('B-Weight')
NERALLLabelMapper.add('B-Railroad')
NERALLLabelMapper.add('I-Unit_Other')
NERALLLabelMapper.add('I-Weight')
NERALLLabelMapper.add('B-Money')
NERALLLabelMapper.add('B-Amphibia')
NERALLLabelMapper.add('I-Mammal')
NERALLLabelMapper.add('I-Company_Group')
NERALLLabelMapper.add('I-Religion')
NERALLLabelMapper.add('I-Temperature')
NERALLLabelMapper.add('B-Stock')
NERALLLabelMapper.add('I-Picture')
NERALLLabelMapper.add('I-GPE_Other')
NERALLLabelMapper.add('B-Show_Organization')
NERALLLabelMapper.add('I-N_Facility')
NERALLLabelMapper.add('I-Animal_Disease')
NERALLLabelMapper.add('B-Art_Other')
NERALLLabelMapper.add('I-Natural_Disaster')
NERALLLabelMapper.add('B-Color_Other')
NERALLLabelMapper.add('I-Insect')
NERALLLabelMapper.add('B-Decoration')
NERALLLabelMapper.add('B-Period_Year')
NERALLLabelMapper.add('B-Living_Thing_Other')
NERALLLabelMapper.add('I-Date')
NERALLLabelMapper.add('I-Political_Party')
NERALLLabelMapper.add('I-Currency')
NERALLLabelMapper.add('B-Intensity')
NERALLLabelMapper.add('B-Organization_Other')
NERALLLabelMapper.add('B-N_Person')
NERALLLabelMapper.add('B-Academic')
NERALLLabelMapper.add('B-Region_Other')
NERALLLabelMapper.add('I-Stock')
NERALLLabelMapper.add('B-Province')
NERALLLabelMapper.add('B-Train')
NERALLLabelMapper.add('B-Rank')
NERALLLabelMapper.add('I-Earthquake')
NERALLLabelMapper.add('B-Day_Of_Week')
NERALLLabelMapper.add('B-Worship_Place')
NERALLLabelMapper.add('B-ID_Number')
NERALLLabelMapper.add('I-Port')
NERALLLabelMapper.add('I-Incident_Other')
NERALLLabelMapper.add('I-Class')
NERALLLabelMapper.add('B-Pro_Sports_Organization')
NERALLLabelMapper.add('B-Incident_Other')
NERALLLabelMapper.add('B-National_Language')
NERALLLabelMapper.add('I-Continental_Region')
NERALLLabelMapper.add('B-Offence')
NERALLLabelMapper.add('I-Car')
NERALLLabelMapper.add('B-Culture')
NERALLLabelMapper.add('B-Picture')
NERALLLabelMapper.add('B-County')
NERALLLabelMapper.add('I-Living_Thing_Part_Other')
NERALLLabelMapper.add('I-Flora_Part')
NERALLLabelMapper.add('B-Flora')
NERALLLabelMapper.add('I-Spaceship')
NERALLLabelMapper.add('B-Tumulus')
NERALLLabelMapper.add('I-Corporation_Other')
NERALLLabelMapper.add('B-Facility_Other')
NERALLLabelMapper.add('B-Bird')
NERALLLabelMapper.add('I-N_Location_Other')
NERALLLabelMapper.add('I-God')
NERALLLabelMapper.add('B-Religion')
NERALLLabelMapper.add('I-N_Product')
NERALLLabelMapper.add('B-Broadcast_Program')
NERALLLabelMapper.add('I-Worship_Place')
NERALLLabelMapper.add('I-Title_Other')
NERALLLabelMapper.add('B-Car')
NERALLLabelMapper.add('I-Star')
NERALLLabelMapper.add('B-Earthquake')
NERALLLabelMapper.add('B-Political_Party')
NERALLLabelMapper.add('I-Canal')
NERALLLabelMapper.add('I-Military')
NERALLLabelMapper.add('B-Magazine')
NERALLLabelMapper.add('I-Latitude_Longtitude')
NERALLLabelMapper.add('B-Mountain')
NERALLLabelMapper.add('I-Tunnel')
NERALLLabelMapper.add('I-N_Person')
NERALLLabelMapper.add('B-Military')
NERALLLabelMapper.add('I-Percent')


POSLabelMapper = Vocabulary(True)
POSLabelMapper.add("O")
POSLabelMapper.add("0")
POSLabelMapper.add("副詞")
POSLabelMapper.add("助詞")
POSLabelMapper.add("動詞")
POSLabelMapper.add("名詞")
POSLabelMapper.add("特殊")
POSLabelMapper.add("判定詞")
POSLabelMapper.add("助動詞")
POSLabelMapper.add("形容詞")
POSLabelMapper.add("感動詞")
POSLabelMapper.add("指示詞")
POSLabelMapper.add("接尾辞")
POSLabelMapper.add("接続詞")
POSLabelMapper.add("接頭辞")
POSLabelMapper.add("連体詞")
POSLabelMapper.add("未定義語")
POSLabelMapper.add("X")
POSLabelMapper.add("[CLS]")
POSLabelMapper.add("[SEP]")

GLOBAL_MAP = {
    'ner': NERLabelMapper,
    'nerall': NERALLLabelMapper,
    'pos': POSLabelMapper,
}

METRIC_META = {
    'ner': [7, 8, 9, 10, 11, 12],
    'nerall': [7, 8, 9, 10, 11, 12],
    'pos': [7, 8, 9, 10, 11, 12],
}

SAN_META = {
    'ner': 2,
    'nerall': 2,
    'pos': 2,
}