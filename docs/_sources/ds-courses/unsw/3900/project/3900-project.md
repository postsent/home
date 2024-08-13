# **Chatbot**

COMP3900 – Computer Science Project

**Project Report**

Team Name: StarPlatinum

University of New South Wales

This project report is submitted on 18.11.2021

# Overview
This project implements a chatbot that provides the service of general query answering with a focus on the medical system and COVID19 support.
## Architecture and Design of software
The software is mainly split into 2 modules – frontend and backend. Frontend is the user interface for our service whereas backend implements the business logics and persists user data.

As the diagram shown below, the frontend will only interact with the server layer for providing services like user authentication and registration, symptom diagnosis and chat history. The conversation with the chatbot begins with the chat endpoint. Frontend data such as user information and chat history will be stored in the MongoDB. After that, the chat messages and context are passed to the Dialogflow gateway to extract keywords and infer the intent of that message. If the intent has a registered handler implemented in our backend, the handler will be called with extracted parameters by Dialogflow and the context by frontend. The handlers are typically requests to third-party APIs or web scrapers for acquiring the necessary data so that we can provide appropriate respond to the users. Finally, the Facebook Messenger and Discord third-party platforms are used for chatbot integration will communicate with Dialogflow Gateway directly for efficiency and it makes the chatbot generic and modular. 

![](3900-architecture.png)

# Software Functionalities
This project aims to create a natural language understanding chatbot that can address the health enquiries of our users. The general usage of this software is to have users ask questions of any concerns they have in the scope of medicine and health. By incorporating Natural Language Understanding, we allow the users to converse with our chatbot non-restrictively, using normal everyday language. Then, our chatbot will process the user’s enquiries and provide an informative response. Frequent users of the chatbot can also have the choice to create an account, allowing the chatbot to hold more useful information about the user so it can give personalised health recommendations. Additionally, our project will include doctors and medical professionals who can provide additional support for the users of our chatbot.

The main features of our software include: (Detailed instruction with images see User manual)
- Customized profile 

The software is built with a user authentication system where users can create an account if they choose to do so. 

Users can use our chatbot in two different modes, guest mode or logged in mode. Guest Users may ask general questions and get a generic response. During the registration of an account, users will have to complete a short survey about their health conditions. This information is stored in our database and can be used by the chatbot to determine a suitable response for the user. Users can also access the Profile page anytime to update their profile. Additionally, all the messages of a logged in user will be saved so the next time a user logs in, they can revisit their chat history.

- General health diagnosis system* 

As normal people, they can only feel the symptoms, it's hard for them to logically link all the	symptoms together and infer the potential risks, due to lack of professional knowledge. Our chatbot can provide GP-level diagnosis result as guidance for patients. Beyond that, there is 24/7 availability which significantly reduces the workload of healthcare workers and gives patients early awareness of their illness.  Patients only need to tell the chatbot about his/her current condition in natural language. The chatbot would utilize the pre-trained machine learning model to analyze the symptoms, determine the possibility of each illness with further inquiries. All the symptoms during the queries would be recorded in patient’s profile, this could help doctors save time during by checking a patient’s health profile, so that they don’t have to waste time doing redundant things.

- COVID19 support 

Nowadays, covid-19 is still a big health risk for our human society. It is highly contagious and leaves the elderly vulnerable. It takes weeks or months to recover after the infection. Because of that, it’s extremely important to prevent people going out if they are potential covid-19 virus carrier. However, many people can’t be aware of that due to the high similarity of covid-19 to normal cold. Our chatbot learns from the online covid-19 database and trains a custom decision tree model to help patients have a better understanding of its infection conditions. It reaches 93% of accuracy in the test sample, which it's enough to provide a guidance for users. User only needs to answer a series of questions, and our chatbot would predict the risk of infection as suggestion. This could help users be aware of their condition and reduce their outdoor activities if they are risky. Besides, we also provide COVID-related information such as the followings:

- COVID19 statistics based on Australia, user can also ask for a particular state covid case number.
- Isolation tips in links format scraped from the NHS UK website. Vaccine information is also given based on the US CDC website.
- COVID-19 hotspot suburb-based information. User can specify a particular state.
- COVID19 symptom diagnosis capability uses will be asked on a series of symptom and the ML model will determine whether the user have caught coronavirus.

- Location-based support 

Information can be useful or unnecessary based on the locations. We want to provide our users with relevant information, hence, determining the location of user becomes a factor to be included. During the start of our chatbot, we ask user for permission to share their location. Location-based features are only available if users agree to sharing their location. Then, the coordinates of the user current location will be sent to our chatbot server. 

With location-based support enabled, our chatbot can find nearby clinics for the user and provide the link to Google Maps where they can have the directions to their desired clinic. For users who are travelling to another country and have health concerns, our chatbot can inform users about the common diseases and some advice on how to avoid them.

- User-friendly interface 

User may have trouble in typing out full sentence or problem in understanding how the chatbot operates, therefore, we introduce features such as shortcut and multiple-choice options to improve accessibility. Besides, the implemented natural language processing engine enables users to express query in standard text form, which makes the interface user-friendly.

- Access to doctor 

User can ask for help from the doctor and leave message to a randomly assigned doctor (we assume registered doctor are all GPs).

The doctor can view patient health information, specify their concerns in text format, their underlying health condition and if there is a record of symptom diagnosis from the chatbot, it will be displayed too.

We assume doctor is at GP level and so when user ask for a doctor to view their request, a doctor will be randomly assigned.

Doctor account has the following access:

- There is a dedicate page for the user to view and modify their profile
- The doctor can view the patient’s health summary
- The doctor can create profile with doctor label
- The doctor leaves their contact and allow user to contact them for further diagnosis

- Personalized health recommendation  

- The recipe recommendation is based on the user health profile. For example, profile that are flagged with overweight would lead to a much lower calories recipe suggestion. This allows us to provide a personalized health service.

- Platform integration 

For the convenience of users, our chatbot is integrated into other messaging platforms. This way, users have more choices to interact with our chatbot. Currently, our chatbot has been integrated to Discord and Facebook Messenger. This demonstrates that the chatbot is generic and modular.

See Jira board for how user stories links to the objective.
# Third-party functionalities

|Licence|Licencing term impact|
| :- | :- |
|MIT|No impact for commercial use and distribution if include the original licence material of the library in the licence of the final product.|
|Proprietary|<p>The free of usage is a subject to change and there is limited number of requests can be made. This may impact the project in terms of long-term maintenance.</p><p>API e.g. its efficiency, complexity may change which means the maintenance of software relies on the quality of the APIs.</p>|
|clause BSD|<p>Low restriction on redistribution </p><p>Open source meaning that the project is maintained by the community and thus free but may not guarantee long-term support</p>|


|Package Name|Uses & Justification|Licence|
| :- | :- | :- |
|React (Facebook Inc, 2021)|React is used for building UI|MIT|
|Redux|Redux is used for holding the app state such as user profile, local chat history|<p>MIT</p><p>	</p>|
|Lodash |Used as utility function library|MIT|
|Material-UI (MUI-org, 2021)|Material-UI is used as components library|MIT|
|Dayjs (iamkun, 2021)|Used for parsing and format dates|MIT|
|Uuid (uuidjs, 2021)|Uuid is used for generating unique id for sessions and chat history|MIT|
|Notistack (iamhosseindhv, 2021)|Notistack is used for sending notifications for feedback for |MIT|
|React-hook-form (react-hook-form, 2021)|It is used for handling form validation such as the sign-up form|MIT|
|React-router-dom (remix-run, 2021)|It is used to provide frontend routing so users can access different pages for various functionalities|MIT|
|React-virtuoso (petyosi, 2021)|Virtualised list for rendering chat history and |MIT|
|Axios (Axios, 2021)|Axios is used for handling requests   |MIT|
|Edamam recipe api|Edamam provides receipt suggestion based on different health tags.  The health profile collects user health info which enables us to give personalised healthy food suggestion.|Proprietary|
|Google Maps|Google Maps API provide the current location to the system as the user request. It helps the system find the nearby clinic.|Proprietary|
|News.api|The news API which provides current news on health category.||
|Dialogflow|The Dialogflow library is used so that we can establish a contact point between our backend and Dialogflow Console. Dialogflow library provides the code to send texts to the Console for Natural Language Understanding and respond to our backend with intents of the text, where our backend will handle the intents accordingly.|Proprietary|
|Discord|Discord API is used to integrate our chatbot to the discord|Proprietary|
|Messenger|Facebook messenger API is used to integrate our chatbot to messenger.|Proprietary|
|Flask|<p>It is easy to pick up and the framework allows us to build up the web application.</p><p>lightweight compared with another populat option such as Django. </p>|clause BSD|
|MongoDB Cloud|No-SQL database allows semi-structured data which is common in natural language and thus is used.|Proprietary|
|BeautifulSoup|It allows us to scrape different websites in a lightweight and efficient way|MIT|
|Infermedica|This gives professional symptom diagnosis with high accuracy and corresponding feedback.|Proprietary|
# Implementation challenges
Covid prediction model 

It is hard to find any available model for the covid-19 prediction online. So, I must find out the data to train the custom model by myself. For any machine learning project, the training data is the key for model. I need search through Internet to find out the data which contain the large amount record data, and its parameters are restricted to the common symptoms. Finally find the appropriate the database in the Kaggle competition website, which contains 3 million recorded patient’s covid diagnosis result and symptoms. The information of the recording data isn’t quite complete. I need to do the feature cleaning and extraction to remove the deficient record, transfer the style for machining learning use. Model choosing is another challenge task, with such large amount of data, I need to balance the training time between accuracy, especially I don’t have especial server to do the training. The model like XGBoost, Navies Bayes are not accessible, even though they might reach good performance. The final decision for the model is the Decision Tree, it takes fair amount of training time and reach 93% accuracy in the test dataset, which is highly acceptable.
## Chat History Rendering
When I was implementing the user interface for chatting with the bot, I realised the issue that as the user chats with the bot, the number of nodes for chat history that are held in DOM will increase and the child nodes of these chat history can sometimes be complicated (e.g.  when rendering a list of items COVID-19). At some point, there would be performance issue. My solution is to render the chat history in a virtualised list implemented by third-party where only the components in the user’s view will be rendered. This method is more memory efficient, and performance can be guaranteed. 

However, all the UI updates are done asynchronously and cannot be access outside the component, which requires other updates, in this case, the auto scroll to bottom feature, to use setTimeout to wait for it. And to avoid memory leak a React useEffect hook will be called to clean up the timeout object.
## Web Scraping 
Website – hard to scrape

Web scraping is the primary method we used to obtain different types of data with no API available. However, some websites are built to prevent such scraping. For example, the website (Coronavirus (COVID-19) case numbers and statistics, 2021) which contains Australia national COVID-19 statistics, has no right click option to view page source. Besides, the table rendered uses JavaScript and fetch it at real time. In some other websites encountered, some purposely uses unstructured page source to prevent user from scraping. Selenium Python library is tried to get over such problem. However, it fails just like BeautifulSoup python library, and it also requires extra setup which is to install 300MB Chrome or Firefox core that enables headless (without GUI) browsing.

After comparing different methods, we decide that, for those that could not be scraped, we will use non-official websites such as <https://covidlive.com.au/> that provides the same information to get over this challenge. Another problem on where to find the website that is appropriate to scrape from. We then find out that different state websites contain links to other state weblink. On top of this, some popular non-official websites provide links to where the resource is based on which allow us to quickly identify the key websites to crawl.

Speed

To improve speed of the scraping, we use SoupStrainer function which is an implementation from Beautifulsoup that enables fast filtering based on the given html tags. Besides, we also use “lxml” and installed the library as mentioned in the official documentation improving performance section (Beautiful Soup Documentation, 2021). Since most endpoints requires single request, multi-thread parallel scraping for multiple requests is not considered. 

Session object is also initialised for requests to allow cookie for future requests which make the scraping more efficient.
## Dialogflow Intention Handler
Since most of our features uses Dialogflow to interpret intentions, we come up with an architecture for backend that uses Dialogflow as a gateway so it can interpret user text input and dispatch the request according to the intention. Each request contains a context that includes information such as location, user ID. Such context will be pass to the registered intention handler.

For the response, it always contains 3 parts. The “type” field acts as a discriminator. The frontend will render the message according to the type. The “message” field contains the response from the bot.  Whereas the “data” field contains extra information such as recipe information that will be rendered alongside the message.

Such architecture follows the Open-Close Principle, which allows us to simply add new feature by writing and registering a new handler. However, Dialogflow cannot handle the symptom descriptions properly due to the complexity of human language, which means we cannot directly add handler to it.

The workaround we used is that users can initiate diagnosis through Dialogflow. Then it will send a response to frontend and the frontend will set a flag in the internal redux store (namely, “chat.diagnosis”) which will then send following requests directly to the backend without Dialogflow. And once the diagnosis is completed, the backend will send back a response with a specific type (namely, “diagnosis/result”) to indicate frontend to unset the flag.
## Others
Flask

We initially consider using Flask. Then we come across Flask-RESTPlus which has full Swagger support and a more modern class implementation for the REST API. Then we find out that it is not maintained anymore but Flask-RESTX, which is a fork version of RESTPlus is currently being maintained by the community and is used in the project.

Integration

For Facebook Messenger chatbot integration, we found that it requires webhook for connection. After some research, there are many popular free choices to expose localhost to a public domain name as listed here in a GitHub collection page (Pitman, 2021). Ngrok and Localtunnel service are the top free choice selected. Localtunnel, however, is unstable during the testing and cannot be validated by Facebook and it only provides free service. Considering that Ngrok has a paid version and shown more reliable during the testing, it is chosen.

Third-party API

Professional third-party API all has a similar API documentation where the data structure is detailed with different endpoint demo. This is one main measurement we use to determine the quality of the API (there are many APIS available, but many are not reliable or powerful). Other considerations are the number of requests quota per month and if free of charge. Besides, to avoid possible downtime caused by the API maintenance, two general symptom check APIs has been implemented. The current using one is Infermedica, where the other – EndlessMedicalAPI being the backup.
# User Documentation / Manual
## Project Setup
The section contains instructions to setup and run the project.  

The chosen system is **Lubuntu.**

It assumes you have the project source code ready in the path project.

Dialogflow account:
-
- Account: xxx
- Password: xxx
- Starplatinum branch
- ![Graphical user interface, text, application, chat or text message Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.002.png)
### Prerequisite
- Nodejs == 14.17.16 # https://github.com/nvm-sh/nvm#installing-and-updating
- Python == 3.8.10
### Folder structure

|<p>project</p><p>├── README.md</p><p>├── backend			# Source code for backend</p><p>├── docker-compose.yml</p><p>├── frontend			# Source code for frontend</p><p>└── workDiaries</p>|
<!-- | :- | -->
### Frontend – Web Interface
**[Prerequisite] Install yarn if you did not have it**

You can install yarn with the following command

|<p>$ npm i -g yarn</p><p>$ yarn –-version #Check installation </p>|
<!-- | :- | -->

**Step 1 – install dependency with** yarn

Please make you are at the project root. If you don’t, please do

|$ cd project/|
<!-- | :- | -->
Then install dependencies with the following commands

|<p>$ cd frontend</p><p>$ yarn </p>|
<!-- | :- | -->

**Step 2 – Start the dev server**

|$ yarn start|
<!-- | :- | -->
Once succeed, you will see the following output:

![Text Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.003.png)
### Backend - Main server
**Step 1 – Create virtual environment**

Please make you are at the project root. If you don’t, please do

|$ cd project/|
<!-- | :- | -->
Then, create and activate virtual environment

|<p>$ cd backend</p><p>$ python3 -m venv ./venv</p>|
<!-- | :- | -->
**Step 2 – Install dependencies**

|<p>$ source ./venv/bin/activate</p><p>(venv) $ python3 -m pip install -r requirements.txt</p>|
<!-- | :- | -->
**Step 3 – Start backend server**

|(venv) python3 ./run.py --cloud # using cloud database|
<!-- | :- | -->

### ML Model Training 

|<p>$ cd backend/covidChecker </p><p>$ bash train.bash</p>|
<!-- | :- | -->

\*Notice the database contain 3 million record data, make sure have enough space and time to train. The existing model is put on the model folder.

### Messenger
**Step 1 – run webhook**

Open a new terminal

|<p>$ cd backend</p><p>$ source ./venv/bin/activate<br>(venv) $ python3 messenger.py # default port 8000</p>|
<!-- | :- | -->

**Step 2 – install and run ngrok**

Open a new terminal

|<p>(venv) $ cd integration</p><p>(venv) $ wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip</p><p>(venv) $ unzip ngrok-stable-linux-amd64.zip</p><p>(venv) $ ./ngrok http 8000 # ur localhost port</p>|
<!-- | :- | -->

**Step 3 – Edit call-back URL**

Copy the https secure URL from the terminal, in below example, it is:  <https://9e4d-129-94-8-155.ngrok.io>

![Text Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.004.png)

Then go to the Facebook developer portal: <https://developers.facebook.com/apps/1486417798399089/messenger/settings/>

and login: 
-
- Account: xxx
- Password: xxx

Now, change the call-back URL to the one previously copied. (In messenger setting section)

Also add the **verify token**: covid19.

![Graphical user interface, text, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.005.png)

**Step 4 – login and test out**

Login a Facebook messenger account and search: **"Health Chat Star"** in the messenger (we created)

Now, you can test out the same features described below except for the general symptom check functionality.

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.006.png)

### Discord 
**[Prerequisite] A Discord account is required to chat with our chatbot.**

**Step 1 – Join chatbot server**

Use the link provided to join the Discord channel: <https://discord.gg/BNJKVJ6q5U> 

![A screenshot of a computer Description automatically generated with medium confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.007.png)

You may need to register for a Discord account if you do not have one.

![A screenshot of a computer Description automatically generated with medium confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.008.png)

Click on “Accept Invite” and you will be brought to the Discord app if you have it installed or click on “Continue to Discord” to go to Discord Web App on your browser.

On Discord, choose the HealthBotServer channel on left panel (should be indicated with a ‘H’) and you should see the channel name HealthBotServer on top left corner. Click on “general” under Text Channels to join the text channel. The chatbot is currently offline as we have not started the client yet.

**Step 2 – Start Discord client**

Open a new terminal

|<p>$ cd backend</p><p>$ source ./venv/bin/activate</p>|
| :- |
|(venv)$ python3 discordbot.py|

You should see the following output:

![Text Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.009.png)

Now, we can go back to Discord and start chatting. You will see HealthBotChatty is online now.

## How to use the chatbot
### User Sign up
A guest user can sign up by clicking the “Register” button in the top right corner to sign into their account.

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.010.png)

Once the button is clicked, they will be redirected to the sign-up page. They need to fill in their personal details to proceed. If the information is incorrect, the border of corresponding text field will turn into red, and an error message will be shown below. Please note, the password requires at least one digit, at least one capital letter, at least one lower case letter, and at least eight characters long. If the user ticks the box of “I’m doctor”, they will be assigned to the role of ‘doctor’, otherwise, the account will be ‘patient’.  Once they have filled the form, they will be directed to the next page to create health profile.

![Graphical user interface, text, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.011.png)

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.012.png)

Users can use the radio buttons to choose their answers for the corresponding questions. All the options are default to False. Users can choose the options that reflect their health conditions. Once finished, they can click finish to log into the system.

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.013.png)
### User Sign In
Users can sign into their account by using the button in the top-right corner.

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.014.png)

Once they are on login page, they can type in their username and password. If they are invalid (wrong email format or does not match the records in the database), the input field will turn red, and an error message will be shown underneath. Once the users signed in successfully, they will be redirected to the home page.

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.015.png) ![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.016.png) 
![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.017.png)
### Modify Profile
Users can modify their profile to update their current health conditions, by clicking the gears button on top right corner and select profile from the dropdown menu.

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.018.png)

This page works in the same way as the profile creation step for signing up. Users can use radio buttons to choose their answer and click finish to update their profile.
### View Chat History
Logged in users can view history by clicking the gears button on top right corner and select profile from the dropdown menu. 

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.019.png)

At History page, they can view all the chat history that is stored on the server, and the history is split into multiple pages. 

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.020.png)
### Requesting Real doctor to diagnose
Registered users can send support tickets describing their health concerns to a real doctor for help by clicking the gear button on the top-right corner and select “New Request” from the menu.

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.021.png)

They can type in their message and click submit to send. However, if they have an active ticket, they cannot send any request until the current ticket is dismissed.

![Graphical user interface, text, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.022.png)![Graphical user interface, text, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.023.png)

If the doctor has replied to a ticket, patients will see the doctor’s message the next time they log in. Then, the ticket will be dismissed.

![Graphical user interface, text, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.024.png)
### View & Reply Support Tickets from Users
Doctors can view support tickets assigned to them by clicking the gears button and choose “Ticket” from the menu.

![Graphical user interface, application Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.025.png)

The table shows a list of tickets assigned to them. Doctors can view and reply to the ticket by clicking the “view” button in each row.

![Graphical user interface, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.026.png)

The ticket detail page shows the detail of the patients including personal information, health conditions and recent diagnosis information. Doctors can type in their message in the “Response” section. And once SUBMIT is clicked the message will be added to the ticket.

![Graphical user interface, text, application, email Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.027.png)
### Shortcut 
Click below bubble to directly make query without typing.

![Graphical user interface, text, application, chat or text message Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.028.png)
### README - Standard Response template
![Graphical user interface, application, Word Description automatically generated](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.029.png)
All responses use a standard UI component, which contains heading, content, source link of the data. In most situation, they are clickable link that redirects the user to the source of that information.


### COVID Statistic (Aus)

|Input|Output|
| :- | :- |
|“Covid case in NSW”, “Covid case in new south wales”|Return COVID statistics given a particular state|
|“Covid case”|Return default location which is Australia nationwide|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.030.png)
### Hotspot

|Input|Output|
| :- | :- |
|“Hotspot in NSW”, “hotspot in new south wales”|<p>Return COVID statistics given a particular state</p><p></p>|
|“Hotspot”|Return default location which is Australia nationwide|
|“hotspot in asdsds”|Return default NSW state hotspot|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.031.png)
### Isolation

|Input|Output|
| :- | :- |
|“I want some isolation tips”|Return COVID statistics given a particular state|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.032.png)Vaccine

|Input|Output|
| :- | :- |
|“I want some vaccine information”|Return three vaccine information – J&J, Pifzer, Moderna based on CDC|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.033.png)
### Travel advice

|Input|Output|
| :- | :- |
|“I want some travel advice to China”|Return country keyword and return travel suggestion e.g. potential health crisis in given location|
|Travel advice to invalid country |Return “Please specify a country and try again”|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.034.png)
### COVID symptom check

|Input|Output|
| :- | :- |
|User: “I want some COVID health check”<br>Bot: “Do you have {this} symptom?”<br>User: “Yes”<br>…<br>Bot “You are diagnosed positive”|After a series of symptom asking, the bot will predict based on the given symptoms whether the user has caught COVID.<br>**Note**<br>During the Q&A, input will be disable.|

![](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.035.png)

![](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.036.png)
### General symptom check

|Input|Output|
| :- | :- |
|<p>User: “I want some symptom check”<br>Bot: “Do you have {this} symptom?”<br>User: “Yes”<br>…<br>Bot: “You are likely to have following risks: <br>COVID-19 <br>0.4416</p><p>… ”</p>|After a series of symptom asking, the bot will predict based on the given symptoms, the disease that the user is most likely to get.<br>The probability is mentioned under each potential result.<br>**Note**<br>During the Q&A, input will be disable.|

![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.037.png)
### Receipt suggestion

|Input|Output|
| :- | :- |
|“I want some recipe suggestion”|Return receipt recommendation based on user’s health profile.<br>Different health tags are used for the food suggestion based on user profile. Calories is the measurement displayed.|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.038.png)
### Health news

|Input|Output|
| :- | :- |
|“I want some health news feed”|Return news feed from trusted source<br>Each links are clickable to redirect other news websites.|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.039.png)
### Nearby clinic

|Input|Output|
| :- | :- |
|“What are some nearby clinic”|Return nearby clinic based on user location if allowed.<br>Each links are clickable to redirect google map with a route from current user location to the destination.<br>After user login, there will be a pop up asking for user location permission. If users give no permission, then a sign of no permission allow will be shown instead.|
![Background pattern Description automatically generated with low confidence](Aspose.Words.5dca8fd7-3572-4136-9b8a-b42b0e958147.040.png)



# Bibliography
Axios. (2021). *Promise based HTTP client for the browser and node.js*. Retrieved 11 17, 2021, from https://github.com/axios/axios

*Beautiful Soup Documentation*. (2021). Retrieved from Beautiful Soup: https://beautiful-soup-4.readthedocs.io/en/latest/#improving-performance

*Coronavirus (COVID-19) case numbers and statistics*. (2021, November). Retrieved from Australian Govenment Department of Health: https://www.health.gov.au/news/health-alerts/novel-coronavirus-2019-ncov-health-alert/coronavirus-covid-19-case-numbers-and-statistics

Facebook Inc. (2021). *A declarative, efficient, and flexible JavaScript library for building user interfaces.* Retrieved Nov 11, 2021, from https://github.com/facebook/react

iamhosseindhv. (2021). *Highly customizable notification snackbars (toasts) that can be stacked on top of each other*. Retrieved 11 17, 2021, from https://github.com/iamhosseindhv/notistack

iamkun. (2021). *Day.js 2kB immutable date-time library alternative to Moment.js with the same modern API*. Retrieved 11 17, 2021, from https://github.com/iamkun/dayjs

Lodash. (2021). *A modern JavaScript utility library delivering modularity, performance, & extras.* Retrieved 11 17, 2021, from https://github.com/lodash/lodash

MUI-org. (2021). *MUI (formerly Material-UI) is the React UI library you always wanted.* Retrieved 11 17, 2021, from https://github.com/mui-org/material-ui

petyosi. (2021). *The most powerful virtual list component for React*. Retrieved 11 17, 2021, from https://github.com/petyosi/react-virtuoso

Pitman, A. (2021). *awesome-tunneling*. Retrieved from GitHub: https://github.com/anderspitman/awesome-tunneling

react-hook-form. (2021). *React Hooks for forms validation (Web + React Native)*. Retrieved 11 17, 2021, from https://github.com/react-hook-form/react-hook-form

reduxjs. (2021). *Predictable state container for JavaScript apps*. Retrieved 11 17, 2021, from https://github.com/reduxjs/redux

remix-run. (2021). *Declarative routing for React*. Retrieved from https://github.com/remix-run/react-router

uuidjs. (2021). *Generate RFC-compliant UUIDs in JavaScript*. Retrieved 11 17, 2021, from https://github.com/uuidjs/uuid

