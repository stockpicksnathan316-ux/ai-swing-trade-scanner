import os
import stripe
from flask import Flask, request, jsonify
from supabase import create_client
from dotenv import load_dotenv  # Add this
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Now get environment variables (they will be loaded from .env)
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
stripe.api_key = os.getenv('STRIPE_API_KEY')

# Check if they exist
if not supabase_url or not supabase_key:
    print("❌ ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not found in .env")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

@app.route('/webhook', methods=['POST'])
def webhook():
    print("🔔 Webhook received!")
    event = request.get_json()
    print(f"Event type: {event.get('type')}")
    
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        email = session['customer_details']['email']
        customer_id = session['customer']
        print(f"📧 Email: {email}")
        print(f"🆔 Customer ID: {customer_id}")
        
        supabase.table('paid_users').upsert({
            'email': email,
            'is_pro': True,
            'stripe_customer_id': customer_id,
            'created_at': datetime.utcnow().isoformat() 
        }).execute()
        
        print(f"✅ Pro access granted to {email}")
        
    elif event['type'] == 'customer.subscription.deleted':
        sub = event['data']['object']
        customer_id = sub['customer']
        supabase.table('paid_users').update({'is_pro': False}).eq('stripe_customer_id', customer_id).execute()
        print(f"❌ Pro access revoked for customer {customer_id}")
    else:
        print(f"ℹ️ Unhandled event: {event['type']}")
    
    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    app.run(port=4242)