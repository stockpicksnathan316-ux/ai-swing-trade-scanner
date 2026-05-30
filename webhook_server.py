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
    
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    # Log partial signature for debugging (first 20 chars)
    if sig_header:
        print(f"📝 Signature (first 20 chars): {sig_header[:20]}...")
    else:
        print("⚠️ No Stripe-Signature header")
    
    # Verify signature
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except ValueError as e:
        print(f"❌ Invalid payload: {e}")
        print(f"Payload preview: {payload[:200]}...")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        print(f"❌ Signature verification failed: {e}")
        print(f"Endpoint secret used (first 10 chars): {endpoint_secret[:10] if endpoint_secret else 'None'}...")
        return jsonify({'error': 'Signature verification failed'}), 400
    
    print(f"✅ Event verified: {event['type']}")
        
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