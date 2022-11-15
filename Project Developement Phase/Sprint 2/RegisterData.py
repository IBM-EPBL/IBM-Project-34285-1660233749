{% extends 'base.html' %}

{% block title %}
Dashboard Page
{% endblock %}


{% block lgNav %}
<a href="{{url_for('profile')}}" class="py-5 px-3 text-blue-800">Profile</a>
<a href="{{url_for('logout')}}"
  class="py-2 px-3 bg-purple-700 text-white hover:bg-purple-300 hover:text-purple-800 rounded transition duration-300 ">Log
  out</a>
{%endblock%}
{% block mdNav%}
<a href="{{url_for('profile')}}" class="block py-2 px-4 text-sm text-blue-800 hover:bg-gray-200">Profile</a>
<a href="{{url_for('logout')}}" class="block py-2 px-4 text-sm hover:bg-gray-200">Log out</a>
{%endblock%}

{% block body %}
<!-- profile card  default block  edit hide -->
<!--BG-->
<div class="flex justify-center items-center h-screen" id="profile">
  <!--Card-->
  <div
    class="bg-purple-600 flex flex-col w-2/3 md:w-1/3 gap-4 py-8 px-6 rounded-xl shadow-xl border   border-purple-400">
    <!--Avatar-->
    <div class="flex justify-between">
      <div><img src="https://www.kindpng.com/picc/m/78-786207_user-avatar-png-user-avatar-icon-png-transparent.png"
          alt="Profile Avatar" class="w-16 h-16 rounded-full shadow-md"></div>
      <div><button class="text-gray-300 hover:text-white" id="editBtn">Update</button></div>
    </div>
    <div class="text-white">
      <!--Role-->
      <p class="text-sm">
        <!-- {% if data['role'] =='' %} -->
              <!-- UI/UX Designer -->
        <!-- {% else %} -->
              {{data['role']}}
        <!-- {% endif %} -->
      </p>
      <!--Name-->
      <p class="font-bold text-2xl">{{data['name']}}</p>
      <!--Phone Number-->
      <p class="text-justify text-sm">{{data['number']}}</p>
      <!--Description-->
      <p class="text-justify text-sm">{{data['email']}}</p>
      <!-- jobfetchcount -->
      <p class="text-justify text-sm">Job Search Count : {{data['jobFetchCount']}}</p>
    </div>

    <!--Tools chip-->
    <div class="flex flex-row gap-1 text-xs text-white font-semibold">
      <div class="rounded-full border border-purple-100 py-1 px-2">
        <span>
          {% if data['skill1'] =='' %}
                Figma
          {% else %}
                {{data['skill1']}}
          {% endif %}
        </span>
      </div>

      <div class="rounded-full border border-purple-100 py-1 px-2">
        <span>
          {% if data['skill2'] =='' %}
                Sketch
          {% else %}
                {{data['skill2']}}
          {% endif %}
        </span>
      </div>

      <div class="rounded-full border border-purple-100 py-1 px-2">
        <span>
          {% if data['skill3'] =='' %}
                Photoshop
          {% else %}
                {{data['skill3']}}
          {% endif %}
        </span>
      </div>
    </div>

  </div>
</div>
<!-- end profile card -->

<!-- profile form default hide edit block -->

<form class="w-11/12 mx-2 lg:mx-80  p-5 m-10 max-w-lg hidden" id="profile-form" action="/update" method="POST">
  <!-- Hide button -->
  <div class="float-right pb-4 text-gray-400 font-bold">
    <button id="hide-button" class="hover:text-gray-600">Hide</button>
  </div>
  <h2 class="text-center pb-4 text-purple-500 font-bold">Update Profile</h2>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="name">
        Full Name
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="cursor-not-allowed bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" disabled id="name" type="text" value="{{data['name']}}">
    </div>
  </div>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class=" block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="email">
        Email Id
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="cursor-not-allowed bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="email" disabled type="text" value="{{data['email']}}">
    </div>
  </div>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="number">
        Phone Number
      </label>
    </div>
    <div class="md:w-2/3">
      <input name="number" class="bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="number" type="text" placeholder="0123456789"  value="{{data['number']}}">
    </div>
  </div>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="role">
        Expecting role
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="role" type="text" name="role" placeholder="UI/UX Designer" value="{{data['role']}}">
    </div>
  </div>

  <div class="block text-gray-500 font-bold  mb-3 pr-4  text-center">skills</div>

  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="skill1">
        Skill-1
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="skill1" type="text" name="skill1" placeholder="Figma" value="{{data['skill1']}}">
    </div>
  </div>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="skill2">
        Skill-2
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="skill2" type="text" name="skill2" placeholder="Sketch" value="{{data['skill2']}}"  >
    </div>
  </div>
  <div class="md:flex md:items-center mb-6">
    <div class="md:w-1/3">
      <label class="block text-gray-500 font-bold md:text-right mb-1 md:mb-0 pr-4" for="skill3">
        Skill-3
      </label>
    </div>
    <div class="md:w-2/3">
      <input class="bg-gray-200 appearance-none border-2 border-gray-200 rounded w-full py-2 px-4 text-gray-700 leading-tight focus:outline-none focus:bg-white focus:border-purple-500" id="skill3" type="text" placeholder="Photoshop" name="skill3" value="{{data['skill3']}}">
    </div>
  </div>
  

  <div class="md:flex md:items-center">
    <div class="md:w-1/3"></div>
    <div class="md:w-2/3">
      <button type="submit" class="shadow bg-purple-500 hover:bg-purple-400 focus:shadow-outline focus:outline-none text-white font-bold py-2 px-4 rounded" type="button">
        Update
      </button>
    </div>
  </div>
</form>


<!-- end profile form   -->
'''
Validation on regitser page
'''
import re 
# re: means regular expression.. used for verify valid emai id 
       
def checkValid(name,email,password):
        error=None
        if name == "":
            error="Username cannot be blank."
        elif len(name) <=2:
            error="Username must be between 3 and 30 characters" 
        if len(password) <6:
            error="Password should contains atleast 6 characters"
        pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.match(pat,email):
            print("Valid Email")
        else:
            error="Email is not valid."
        return error
{% block js %}
<script src="{{url_for('static', filename='profile.js')}}"></script>
{%endblock%}
<!-- profile card end -->
{% endblock%}