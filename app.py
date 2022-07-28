from pycaret.regression import load_model, predict_model
import streamlit as st
from readline import set_pre_input_hook
from sys import setprofile
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import plotly.express as px







def predict_survival(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]



model = load_model('deaths')


st.title('COVID gravity prediction')
st.write('This is a predictor of risk incurred when getting COVID 19')

range_age = st.sidebar.selectbox('Age',('0-17','18-49','50-64','65+'))
age = 0 
if range_age == "0-17":
    age = 0

elif range_age == "18-49":
    age = 1
elif range_age == "50-64":
    age = 2
elif range_age == "65+":
    age = 3

    
sex_options = st.sidebar.selectbox('Sex',('Male','Female'))
sex = 1
if sex_options == "Male":
    sex = 0
else:
    sex = 1

state_name = st.sidebar.selectbox('State',('SC', 'MN', 'NY', 'AL', 'IA', 'MD', 'TN', 'CO', 'IN', 'AR', 'KY',
       'WI', 'NM', 'MS', 'CA', 'VA', 'MA', 'NE', 'NC', 'LA', 'NH', 'OK',
       'MO', 'OH', 'NJ', 'GA', 'UT', 'KS', 'IL', 'MI', 'DE', 'FL', 'SD',
       'TX', 'WA', 'WY', 'PA', 'AK', 'ND', 'OR', 'ME', 'CT', 'AZ', 'ID',
       'NV', 'WV', 'HI', 'VT', 'MT', 'DC', 'PR', 'GU'))


county = st.sidebar.selectbox('County',('RICHLAND', 'STEARNS', 'NIAGARA', 'FRANKLIN', 'SIOUX',
       'ANNE ARUNDEL', 'WILSON', 'WELD', 'ST. JOSEPH', 'BENTON', 'KENTON',
       'RACINE', 'HOWARD', 'MCKINLEY', 'JONES', 'LOS ANGELES', 'ONEIDA',
       'ALLEN', 'SHENANDOAH', 'NORFOLK', 'LOUDOUN', 'RANKIN', 'DAWSON',
       'MANASSAS CITY', 'SURRY', 'PLATTE', 'MACON', 'MADISON',
       'MUSCATINE', 'PLAQUEMINES', 'RANDOLPH', 'BOONE', 'LOGAN', 'DODGE',
       'KINGS', 'ROCKINGHAM', 'SCOTT', 'HARRISON', 'GASTON', 'GREENE',
       'CARROLL', 'DOOR', 'HILLSBOROUGH', 'SONOMA', 'NESHOBA', 'CARTER',
       'ST. LOUIS CITY', 'SHELBY', 'ST. LAWRENCE', 'ULSTER', 'ASHTABULA',
       'FAUQUIER', 'SULLIVAN', 'FREDERICK', 'ALLEGANY', 'CUMBERLAND',
       'PORTSMOUTH CITY', 'WAYNE', 'MARIN', 'ASHE', 'STEELE', 'MORROW',
       'NELSON', 'CATTARAUGUS', 'FULTON', 'BOX ELDER', 'WYANDOTTE',
       'CHATHAM', 'MERCER', 'HAYWOOD', 'ADAIR', 'GRAYSON', 'STAFFORD',
       'GRUNDY', 'CAMDEN', 'DUBUQUE', 'DAVIDSON', 'BRANCH', 'MUSCOGEE',
       'CAYUGA', 'JASPER', 'TIOGA', 'LEE', 'SAN BENITO', 'SUMMIT',
       'IBERIA', 'DARKE', 'BROWN', 'MOWER', 'BARTHOLOMEW', 'ST. TAMMANY',
       'ROBESON', 'MURRAY', 'MARION', 'BEDFORD', 'WEBSTER', 'COMAL',
       'DINWIDDIE', 'CHAMBERS', 'LENOIR', 'TAYLOR', 'VIGO', 'WORCESTER',
       'GARFIELD', 'KALAMAZOO', 'DURHAM', 'WASHINGTON', 'PIERCE',
       'CIBOLA', 'TROUP', 'FREMONT', 'CENTRE', 'HURON', 'KANE',
       'KENAI PENINSULA', 'KANDIYOHI', 'WHITFIELD', 'MCCURTAIN', 'UNION',
       'RICHMOND', 'PRINCE GEORGE', 'COLUMBIA', 'CATAWBA', 'COMANCHE',
       'OCEANA', 'SANTA ROSA', 'TOMPKINS', 'NEW HANOVER', 'BUTLER',
       'FREEBORN', 'GREEN', 'ORANGE', 'LOUISA', 'PUEBLO', 'DE SOTO',
       'DELAWARE', 'SCHENECTADY', 'WILKES', 'BLACK HAWK', 'DESOTO',
       'CROW WING', 'NEWBERRY', 'COWETA', 'PEACH', 'HANOVER',
       'LAUDERDALE', 'ROCKLAND', 'NICOLLET', 'FORREST', 'PHILADELPHIA',
       'FAYETTE', 'BEAVER', 'DAKOTA', 'TUSCARAWAS', 'CASS', 'ACCOMACK',
       'BARNSTABLE', 'FAIRFAX CITY', 'BRYAN', 'MERIWETHER', 'JOHNSON',
       'OGLE', 'SANTA CRUZ', 'CREEK', 'CHESTERFIELD', 'FINNEY', 'WAPELLO',
       'CAPE MAY', 'SUMTER', 'LAKE', 'HARDIN', 'CULLMAN', 'LIVINGSTON',
       'OTTER TAIL', 'JACKSON', 'MAHASKA', 'ORANGEBURG', 'CLACKAMAS',
       'SUSSEX', 'LINN', 'SALINE', 'GENESEE', 'WEST BATON ROUGE',
       'PENNINGTON', 'SALEM', 'WASATCH', 'NORTHUMBERLAND', 'HOOD RIVER',
       'MOREHOUSE', 'BARTOW', 'CRAVEN', 'SMITH', 'PAGE', 'STANLY',
       'PASQUOTANK', 'LINCOLN', 'LENAWEE', 'SALUDA', 'EFFINGHAM',
       'BARBOUR', 'ANDROSCOGGIN', 'SENECA', 'CRAIGHEAD', 'CLEVELAND',
       'WINDHAM', 'LASALLE', 'WALKER', 'RENSSELAER', 'LANCASTER', 'HENRY',
       'ELBERT', 'CALHOUN', 'ANOKA', 'WALLER', 'ESSEX', 'ROCK', 'DALLAS',
       'ELKHART', 'KENOSHA', 'HAMPDEN', 'MIAMI', 'MIDDLESEX', 'JEFFERSON',
       'LAFAYETTE', 'SAN LUIS OBISPO', 'TITUS', 'GILA', 'YATES',
       'CRAWFORD', 'HARNETT', 'FAIRFAX', 'PUTNAM', 'JEROME', 'CLAY',
       'DORCHESTER', 'BUCHANAN', 'COOK', 'DEKALB', 'OTERO', 'LEAVENWORTH',
       'AIKEN', 'CALUMET', 'BERRIEN', 'HERKIMER', 'CATOOSA', 'AUDRAIN',
       'TODD', 'SEWARD', 'CADDO', 'VICTORIA', 'IOSCO', 'MONTGOMERY',
       'PRINCE EDWARD', 'YORK', 'BLUE EARTH', 'CASWELL', 'SUFFOLK',
       'ALEXANDRIA CITY', 'WARREN', 'WOOD', 'TOOELE', 'LYON', 'MAURY',
       'MECKLENBURG', 'ALACHUA', 'WOODBURY', 'MOHAVE', 'MINIDOKA',
       'DECATUR', 'BOULDER', 'BURLINGTON', 'MARSHALL', 'TULSA', 'BEE',
       'TUSCALOOSA', 'COFFEE', 'PALM BEACH', 'GREGG', 'ETOWAH',
       'EAST BATON ROUGE', 'CAROLINE', 'MERRIMACK', 'SCOTTS BLUFF',
       'DOUGLAS', 'GRAY', 'LUZERNE', 'PLYMOUTH', 'WYOMING', 'IMPERIAL',
       'HOUSTON', 'MENDOCINO', 'ST. CLAIR', 'ALPENA', 'PULASKI',
       'HAMPSHIRE', 'MONTEZUMA', 'KITTITAS', 'MITCHELL', 'DILLON',
       'DICKSON', 'EAU CLAIRE', 'CLINTON', 'GONZALES', 'KENT', 'HORRY',
       'RUSSELL', 'NEW HAVEN', 'ALEXANDER', 'OKALOOSA', 'BUNCOMBE',
       'YELL', 'JO DAVIESS', 'KANAWHA', 'WICOMICO', 'BERKSHIRE', 'UPSON',
       'HALL', 'MOBILE', 'HOLMES', 'YAKIMA', 'BREMER', 'DES MOINES',
       'HOCKING', 'BURKE', 'BELKNAP', 'WHITLEY', 'FORSYTH', 'AVOYELLES',
       'SAUNDERS', 'RAMSEY', 'DAUPHIN', 'SANGAMON', 'HAMILTON',
       'WASHTENAW', 'RICE', 'ARLINGTON', 'KNOX', 'SAMPSON', 'LAGRANGE',
       'IRON', 'CHAUTAUQUA', 'ALAMANCE', 'ISLE OF WIGHT', 'LUCAS', 'HOKE',
       'SUWANNEE', 'WESTCHESTER', 'FAIRFIELD', 'BAY', 'ONTARIO', 'OTSEGO',
       'LEWIS', 'ALLEGAN', 'PASCO', 'BROOME', 'ROCKDALE', 'MADERA',
       'MUHLENBERG', 'POPE', 'TIPTON', 'CHISAGO', 'CHARLESTON',
       'WILLIAMSBURG', 'MCCLAIN', 'POTTAWATTAMIE', 'HUNTERDON',
       'UMATILLA', 'BANNOCK', 'GEORGETOWN', 'ST. CHARLES',
       'BALTIMORE CITY', 'TALBOT', 'CALLOWAY', 'POLK', 'TALLAPOOSA',
       'MARTIN', 'COLQUITT', 'CULPEPER', 'PETERSBURG CITY', 'BLAINE',
       'IBERVILLE', 'STODDARD', 'FANNIN', 'DENVER', 'WARE', 'JESSAMINE',
       'CLARKE', 'MCDUFFIE', 'HOT SPRING', 'POTTER', 'LOUDON', 'SEVIER',
       'GUILFORD', 'VOLUSIA', 'YAVAPAI', 'MACOUPIN', 'MILWAUKEE',
       'PORTAGE', 'CRISP', 'SUFFOLK CITY', 'PERSON', 'YOLO',
       'TREMPEALEAU', 'WISE', 'QUEENS', 'WYANDOT', 'COWLEY', 'CHEROKEE',
       'FLOYD', 'RIO ARRIBA', 'TOLLAND', 'LUMPKIN', 'OHIO', 'ATHENS',
       'TATTNALL', 'STARK', 'BIBB', 'LAWRENCE', 'WALWORTH', 'CORYELL',
       'GENEVA', 'MONROE', 'DUPAGE', 'HINDS', 'HARDEMAN',
       'PRINCE WILLIAM', 'LEBANON', 'DARLINGTON', 'HARDEE', 'NOBLES',
       'COVINGTON', 'STEUBEN', 'HERTFORD', 'HENNEPIN', 'NAVAJO',
       'GUADALUPE', 'ST. CROIX', 'CLARENDON', 'MORRISON', 'LAPEER',
       'COLORADO', 'JAMES CITY', 'MONTEREY', 'PENOBSCOT', 'KOSCIUSKO',
       'NEW CASTLE', 'ALLEGHENY', 'AUTAUGA', 'EVANGELINE', 'NEWAYGO',
       'ARAPAHOE', 'BARTON', 'CLAYTON', 'MILLER', 'RAPIDES', 'DEL NORTE',
       'ASSUMPTION', 'SCHUYLKILL', 'KAUFMAN', 'FRESNO', 'BELMONT',
       'TERREBONNE', 'MOORE', 'OLDHAM', 'EL PASO', 'KING', 'PIKE',
       'CARLTON', 'LAURENS', 'LIBERTY', 'CLATSOP', 'LYCOMING', 'CASSIA',
       'OUACHITA', 'CHEMUNG', 'UVALDE', 'HOPEWELL CITY', 'HUNT', 'MORGAN',
       'THOMAS', 'CARBON', 'GRIMES', 'MILLE LACS', 'KLEBERG', 'STARR',
       'GRAFTON', 'MARLBORO', 'WHITE', 'BALDWIN', 'COLLETON', 'PAULDING',
       'FOND DU LAC', 'PINELLAS', 'SAN DIEGO', 'BOSSIER', 'CLARK',
       'KANKAKEE', 'HOCKLEY', 'STOKES', 'LIMESTONE', 'SAN JOAQUIN',
       'DENTON', 'HUDSON', 'FORD', 'GILMER', 'NEZ PERCE', 'CHEATHAM',
       'ROBERTSON', 'MULTNOMAH', 'DANVILLE CITY', 'WILL', 'OUTAGAMIE',
       'COLUMBUS', 'GRANVILLE', 'DUNKLIN', 'COLUMBIANA', 'NEWTON',
       'BARRON', 'PANOLA', 'GRANT', 'NORTHAMPTON', 'VANCE', 'WILLIAMSON',
       'FLUVANNA', 'GREENWOOD', 'CHENANGO', 'NEWPORT NEWS CITY',
       'HIDALGO', 'WASCO', 'SOMERSET', 'BEXAR', 'MANATEE', 'LACKAWANNA',
       'JAY', 'SEMINOLE', 'GIBSON', 'DUTCHESS', 'DALE', 'NASH', 'OSWEGO',
       'WINCHESTER CITY', 'CERRO GORDO', 'CALDWELL', 'KENNEBEC',
       'WINSTON', 'ADAMS', 'HART', 'SPOTSYLVANIA', 'RUTHERFORD',
       'HARRISONBURG CITY', 'RHEA', 'LEHIGH', 'CACHE', 'GEAUGA',
       'SALT LAKE', 'SANDUSKY', 'OTTAWA', 'SARPY', 'MORRIS', 'HABERSHAM',
       'CHESAPEAKE CITY', 'OCEAN', 'MCHENRY', 'MARATHON', 'AUGLAIZE',
       'RICHMOND CITY', 'POTTAWATOMIE', 'WRIGHT', 'HARTFORD', 'GADSDEN',
       'CUYAHOGA', 'SANTA BARBARA', 'HENDRY', 'HUNTINGDON', 'DUPLIN',
       'COBB', 'MCMINN', 'MONTROSE', 'LAVACA', 'BROOMFIELD', 'GLYNN',
       'CECIL', 'DOUGHERTY', 'HENRICO', 'BERGEN', 'WEBB', 'CHESTER',
       'TOM GREEN', 'GRADY', 'NEW KENT', 'PERRY', 'SCHOHARIE', 'LICKING',
       'KLICKITAT', 'WINNEBAGO', 'SAGINAW', 'SPENCER', 'WABASH',
       'STEPHENS', 'ALBANY', 'GRATIOT', 'GOOCHLAND', 'ANDERSON', 'HALE',
       'HANCOCK', 'STEPHENSON', 'ACADIA', 'HEMPSTEAD', 'DUNN', 'BELL',
       'KEWAUNEE', 'MEEKER', 'WILLIAMS', 'BUTTS', 'LOWNDES', 'VAN ZANDT',
       'ST. MARY', 'APACHE', 'MCDONOUGH', 'WAGONER', 'FRIO', 'WORTH',
       'COLES', 'CHELAN', 'POINTE COUPEE', 'OKEECHOBEE', 'OCONTO',
       'COLLIER', 'DAVIESS', 'STARKE', 'BUCKS', 'KERSHAW', 'AMHERST',
       'ROCKWALL', 'KING GEORGE', 'ST. MARTIN', 'ST. LOUIS', 'HENDERSON',
       'LAFOURCHE', 'AUGUSTA', 'PICKAWAY', 'MCLEOD', 'DYER', 'HICKMAN',
       "ST. MARY'S", 'INDEPENDENCE', 'LITCHFIELD', 'GORDON', 'LUBBOCK',
       'LORAIN', 'BRAZORIA', 'CHARLOTTE', 'BARREN', 'FREDERICKSBURG CITY',
       'SHAWNEE', 'MALHEUR', 'BARNWELL', 'MUSKINGUM', 'HARRIS',
       'HUTCHINSON', 'ERIE', 'SARATOGA', 'BEAUREGARD', 'SHERBURNE',
       'IONIA', 'MEDINA', 'MCCRACKEN', 'CURRY', 'JERSEY', 'DO�A ANA',
       'CHARLES', 'HALIFAX', "QUEEN ANNE'S", 'ST. BERNARD', 'BELTRAMI',
       'COCONINO', 'ROANE', 'GLENN', 'VAN BUREN', 'CRITTENDEN',
       'WHITESIDE', 'ATASCOSA', 'ATLANTIC', 'KENDALL', 'NOBLE', 'STORY',
       'WAYNESBORO CITY', 'ALBEMARLE', 'WARD', 'PINE', 'MCDONALD',
       'NACOGDOCHES', 'TULARE', 'HONOLULU', 'CORTLAND', 'MEADE', 'WEBER',
       'SABINE', 'SAN MATEO', 'YAMHILL', 'OXFORD', 'CARVER', 'NUECES',
       'OKANOGAN', 'ITASCA', 'COSHOCTON', 'CALLAWAY', 'FORT BEND',
       'VIRGINIA BEACH CITY', 'ABBEVILLE', 'SANDOVAL', 'LE SUEUR',
       'LYNCHBURG CITY', 'YUMA', 'YANKTON', 'DAVIE', 'KERN', 'GRAVES',
       'OSAGE', 'ORLEANS', 'HAMPTON CITY', 'TIFT', 'BERKS', 'YADKIN',
       'STAUNTON CITY', 'WALTON', 'BASTROP', 'SNOHOMISH', 'TUSCOLA',
       'WATAUGA', 'DUBOIS', 'RAY', 'VERMILION', 'BOYLE', 'PICKENS',
       'POWHATAN', 'LARIMER', 'OSCEOLA', 'MORTON', 'OZAUKEE', 'ASCENSION',
       'GLOUCESTER', 'OCONEE', 'CHATTOOGA', 'SKAGIT', 'DANE', 'CALCASIEU',
       'IROQUOIS', 'PORTER', 'BUFFALO', 'VERNON', 'LEON', 'MATAGORDA',
       'STANISLAUS', 'MERCED', 'NATCHITOCHES', 'BALTIMORE', 'BROWARD',
       'VANDERBURGH', 'WALLA WALLA', 'MINNEHAHA', 'CALVERT', 'CHESHIRE',
       'PETTIS', 'STRAFFORD', 'ST. JAMES', 'CHRISTIAN', 'MCLEAN',
       'LAPORTE', 'BULLOCH', 'WASHOE', 'WAKULLA', 'HILLSDALE', 'POINSETT',
       'ISANTI', 'CUSTER', 'MISSISSIPPI', 'EDGECOMBE', 'SACRAMENTO',
       'NAVARRO', 'SEDGWICK', 'HAYS', 'ST. JOHN THE BAPTIST', 'WILLACY',
       'ANGELINA', 'IREDELL', 'PEORIA', 'SAN MIGUEL', 'DELTA', 'WHARTON',
       'SUTTER', 'BRISTOL', 'TANGIPAHOA', 'INGHAM', 'OLMSTED', 'TRUMBULL',
       'MIAMI-DADE', 'BARROW', 'ESCAMBIA', 'CANADIAN', 'TWIN FALLS',
       'HENDRICKS', 'ANSON', 'OVERTON', 'EATON', 'FLAGLER', 'ELLIS',
       'WOODFORD', 'WAUPACA', 'WYTHE', 'BRONX', 'BRADLEY', 'RUSK',
       'RIPLEY', 'PIMA', 'ST. LANDRY', 'CHITTENDEN', 'ROANOKE CITY',
       'HUNTINGTON', 'PINAL', 'GRAHAM', 'ONONDAGA', 'TAOS', 'COLBERT',
       'MASON', 'SAN JUAN', 'TOOMBS', "PRINCE GEORGE'S",
       'JEFFERSON DAVIS', 'MUSKOGEE', 'JIM WELLS', 'ST. LUCIE', 'PENDER',
       'BLADEN', 'HUMBOLDT', 'CAPE GIRARDEAU', 'WELLS', 'JENNINGS', 'NYE',
       'BLOUNT', 'WAUKESHA', 'EDGEFIELD', 'GRAND FORKS', 'GUERNSEY',
       'BARRY', 'ELMORE', 'SALEM CITY', 'SPALDING', 'GALVESTON',
       'CHARLOTTESVILLE CITY', 'BOWIE', 'STUTSMAN', 'CODINGTON',
       'MIDLAND', 'LEXINGTON', 'NEW LONDON', 'HARFORD', 'BINGHAM', 'NAPA',
       'WHATCOM', 'SHAWANO', 'OAKLAND', 'CARSON CITY', 'SNYDER', 'PITT',
       'PITTSYLVANIA', 'CHILTON', 'SUSQUEHANNA', 'CABARRUS', 'EAGLE',
       'OBION', 'SWEETWATER', 'BEAUFORT', 'SHIAWASSEE', 'SCOTLAND', 'ADA',
       'MCDOWELL', 'LEA', 'DEARBORN', 'BRAZOS', 'GARLAND', 'LEVY',
       'GOODHUE', 'YUBA', 'PASSAIC', 'OKLAHOMA', 'PLACER', 'RANDALL',
       'ROCK ISLAND', 'SUMNER', 'ARMSTRONG', 'COCHISE', 'THURSTON',
       'CHAVES', 'NASSAU', 'INDIAN RIVER', 'WARRICK', 'NATRONA', 'RILEY',
       'WAKE', 'TALLADEGA', 'FLORENCE', 'EDDY', 'BECKER', 'MARINETTE',
       'CHAMPAIGN', 'NORFOLK CITY', 'TIPPECANOE', 'MUSKEGON',
       'MATANUSKA-SUSITNA', 'ISABELLA', 'VALENCIA', 'MIFFLIN', 'ROWAN',
       'MONMOUTH', 'SAUK', 'HIGHLANDS', 'BULLITT', 'MCLENNAN', 'ROANOKE',
       'CLEARFIELD', 'CHIPPEWA', 'HIGHLAND', 'TRAVIS', 'LARAMIE',
       'CANYON', 'SPARTANBURG', 'SHEBOYGAN', 'MONTCALM', 'ELKO',
       'SANTA FE', 'JOHNSTON', 'LA PLATA', 'ALAMEDA', 'ROSS', 'NEW YORK',
       'ROGERS', 'MAUI', 'BERKELEY', 'ECTOR', 'INDIANA', 'TAZEWELL',
       'CARTERET', 'SEBASTIAN', 'GWINNETT', 'CITRUS', 'WESTMORELAND',
       'MANITOWOC', 'LA CROSSE', 'LONOKE', 'HAWAII', 'PARKER', 'BUTTE',
       'BRUNSWICK', 'ST. JOHNS', 'RIVERSIDE', 'DAVIS', 'BRADFORD', 'RENO',
       'EL DORADO', 'SOLANO', 'SHASTA', 'MACOMB', 'CLERMONT', 'DESCHUTES',
       'WICHITA', 'BLAIR', 'COLLIN', 'SAN FRANCISCO', 'ST. FRANCOIS',
       'GALLATIN', 'CAMERON', 'CONTRA COSTA', 'GREENVILLE', 'HAMBLEN',
       'HERNANDO', 'MAHONING', 'SARASOTA', 'SAN BERNARDINO',
       'YELLOWSTONE', 'BURLEIGH', 'COWLITZ', 'VENTURA', 'BONNEVILLE',
       'ONSLOW', 'FAULKNER', 'MARICOPA', 'SPOKANE', 'ANCHORAGE',
       'CAMBRIA', 'BERNALILLO', 'KITSAP', 'BREVARD', 'DUVAL',
       'SANTA CLARA', 'LANE', 'KOOTENAI', 'UTAH', 'TARRANT'))

race = st.sidebar.selectbox('Race',('White',
       'American Indian/Alaska Native', 'Black', 'Asian',
       'Multiple/Other', 'Native Hawaiian/Other Pacific Islander'))


features = {'age':age, 'sex':sex, 'state':state, 'race':race}


df = pd.DataFrame(features, index = [0])
prediction = predict_survival(model, df)       
features_df  = pd.DataFrame([features])




st.table(features_df)

if st.button('Predict'):    
    prediction = predict_survival(model, features_df)
    if prediction == True:
        st.write('The risk to your safety is high. Please take precautions and try avoiding infection'+ str(prediction))
    else:
        st.write('The risk is not as high. Please help ensure safety to those vulnerable around you.')
