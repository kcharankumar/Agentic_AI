
#LLM Integration will be done in the next program.
# defining the functions are like tools.  We use tools to perform specific tasks. 
# In this program we are defining the tools and then using them in the main program.
# We are not using any LLM in this program.  We will use LLM in the next program.
# Integrating LLM and human tools will make the program complete to an extent.
# This program has only python functions.  As the first step, as part of tools,
# we are defining the functions to search for restaurants, show menu, place order and make payment.

def search_restaurant():
    print("Searching for restaurants...")
    print ("1. Sonam Restaurant")
    print ("2. Paneer house..")
    
def show_menu():
    print("\nShowing menu...")
    print ("1. Paneer Butter Masala")
    print ("2. Dal Makhani")
    print ("3. Veg Biryani")
    
def place_order():
    print("\nPlacing order...")
    print ("Order placed successfully!")
    
def make_payment():
    print("\nMaking payment...")
    print ("Payment successful!")
    
#Get prompt from user
prompt = input("Which type of restaurant you want to search for? (e.g., Indian, Chinese, Italian): ")
print ('You have selected:', prompt)

search_restaurant()
show_menu()
place_order()
make_payment()
