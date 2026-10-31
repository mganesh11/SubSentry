import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta
import os

# ==============================================================================
# 1. CONSTANTS & SERVICE RECOGNITION DATABASE
# ==============================================================================
KNOWN_SERVICES = [
    (r'NETFLIX', 'Netflix', 'Streaming & Video', 'Log into netflix.com > Account > Membership & Billing > Cancel Membership. Confirm cancellation on screen.'),
    (r'SPOTIFY', 'Spotify', 'Music & Audio', 'Visit spotify.com/account > Under Your Plan, click Change Plan > Scroll down to Cancel Spotify > Cancel Premium.'),
    (r'YOUTUBE', 'YouTube Premium', 'Streaming & Video', 'Open YouTube app > Tap Profile > Purchases and memberships > Select YouTube Premium > Deactivate > Cancel.'),
    (r'GOOGLE.*(STORAGE|ONE|GSUITE|DRIVE)', 'Google One (Storage)', 'Cloud & Productivity', 'Go to one.google.com > Settings gear icon > Change membership plan > Cancel membership.'),
    (r'SWIGGY', 'Swiggy One', 'Food & Dining Pass', 'Open Swiggy app > Profile > Swiggy One Membership > Manage Membership > Opt out / Cancel renewal.'),
    (r'ZOMATO', 'Zomato Gold', 'Food & Dining Pass', 'Open Zomato app > Profile > Zomato Gold > Manage Membership > Turn off Auto-Renewal.'),
    (r'AMAZON.*PRIME', 'Amazon Prime', 'E-commerce & Streaming', 'Go to amazon.in > Account & Lists > Prime Membership > Manage Membership > End Membership.'),
    (r'THE KEN', 'The Ken Newsletter', 'News & Media', 'Log into the-ken.com > My Account > Subscriptions > Turn off Auto-Renew.'),
    (r'DISNEY|HOTSTAR', 'Disney+ Hotstar', 'Streaming & Video', 'Open Hotstar > Help & Settings > Subscriptions > Cancel Subscription.'),
    (r'APPLE|ICLOUD', 'Apple Services (iCloud/Music)', 'Cloud & Storage', 'iOS Settings > Tap Name > Subscriptions > Select Subscription > Cancel Subscription.'),
    (r'MICROSOFT|OFFICE\s*365', 'Microsoft 365', 'Productivity', 'Visit account.microsoft.com/services > Manage > Cancel subscription.'),
    (r'CULT\.?FIT|GYM|FITNESS', 'Cult.fit Fitness Pass', 'Health & Fitness', 'Open Cult.fit app > Profile > Active Packs > Manage / Pause Membership.'),
    (r'CHATGPT|OPENAI', 'ChatGPT Plus', 'AI & Productivity', 'Visit chatgpt.com > Settings > Subscription > Manage > Cancel Plan.'),
]

# ==============================================================================
# 2. CORE RECURRING DETECTION & AUDITING ENGINE
# ==============================================================================

def normalize_merchant(description: str):
    """Extract clean merchant name, category, and standard cancellation guide."""
    d_upper = str(description).upper()
    for pattern, name, cat, guide in KNOWN_SERVICES:
        if re.search(pattern, d_upper):
            return name, cat, guide
            
    # Generic normalization for unspecified merchants
    cleaned = re.sub(r'^(UPI-|POS |ACH DEBIT - |ACH |NEFT |IMPS |BILLDESK |RAZORPAY\*?)', '', d_upper)
    cleaned = re.sub(r'-(AUTOPAY|E-MANDATE|MUMBAI|BLR|BAN|BANGALORE|DELHI|CHENNAI|KORAMANGALA|WHITEFIELD).*', '', cleaned)
    cleaned = re.sub(r'\b(PVT|LTD|CORP|INDIA|PAYMENTS|RECURRING|MEMBERSHIP|ANNUAL|MONTHLY|QUARTERLY|WEEKLY)\b', '', cleaned)
    cleaned = re.sub(r'[^A-Z0-9\s]', ' ', cleaned).strip()
    words = cleaned.split()
    name = ' '.join(words[:3]).title() if words else str(description)[:20].title()
    guide = f"Log into the merchant portal or mobile app for {name}, navigate to Account Settings > Subscriptions/Billing, and choose Cancel Plan."
    return name, 'General / Utilities', guide


def clean_and_validate_csv(raw_df: pd.DataFrame):
    """
    Validates CSV, matches flexible column names, handles invalid dates and negative amounts,
    and returns a cleaned DataFrame.
    """
    df = raw_df.copy()
    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if any(k in c_lower for k in ['date', 'txn_dt', 'time', 'timestamp']):
            col_map[col] = 'date'
        elif any(k in c_lower for k in ['desc', 'narrat', 'detail', 'merchant', 'particular']):
            col_map[col] = 'description'
        elif any(k in c_lower for k in ['amount', 'debit', 'amt', 'withdraw', 'spend', 'cost']):
            col_map[col] = 'amount'
            
    df = df.rename(columns=col_map)
    required = ['date', 'description', 'amount']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Please provide a CSV with date, description, and amount.")
        
    # Robust date parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    invalid_dates_count = int(df['date'].isna().sum())
    df = df.dropna(subset=['date'])
    
    # Amount parsing (strips currency symbols, commas, handles negative debit signs)
    if df['amount'].dtype == object:
        df['amount'] = df['amount'].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    invalid_amounts_count = int(df['amount'].isna().sum())
    df = df.dropna(subset=['amount'])
    
    # In bank statements, debit can be negative or positive; treat debit magnitude as expense
    df['amount'] = df['amount'].abs()
    df = df[df['amount'] > 0]
    
    df['description'] = df['description'].fillna('Unknown Transaction').astype(str).str.strip()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Apply merchant normalization
    norm_results = df['description'].apply(normalize_merchant)
    df['merchant'] = [r[0] for r in norm_results]
    df['category'] = [r[1] for r in norm_results]
    df['cancel_guide'] = [r[2] for r in norm_results]
    
    return df, invalid_dates_count, invalid_amounts_count


def classify_cadence(mean_interval: float, std_interval: float):
    """Identifies billing cadence based on mean days and interval variance."""
    if 5.0 <= mean_interval <= 10.0 and std_interval <= 3.5:
        return 'Weekly'
    elif 11.0 <= mean_interval <= 18.0 and std_interval <= 4.5:
        return 'Bi-Weekly'
    elif 24.0 <= mean_interval <= 37.0 and std_interval <= 6.5:
        return 'Monthly'
    elif 75.0 <= mean_interval <= 105.0 and std_interval <= 12.0:
        return 'Quarterly'
    elif 165.0 <= mean_interval <= 205.0 and std_interval <= 16.0:
        return 'Semi-Annual'
    elif 335.0 <= mean_interval <= 395.0 and std_interval <= 25.0:
        return 'Annual'
    return None


def detect_subscriptions(df: pd.DataFrame):
    """
    Core detection algorithm:
    1. Groups transactions by normalized merchant.
    2. Enforces minimum 2 occurrences (never classifies 1 occurrence as recurring).
    3. Calculates intervals and interval stability.
    4. Distinguishes fixed/step subscriptions from erratic variable spending.
    5. Calculates normalized monthly and annual costs.
    6. Predicts next upcoming renewal date.
    7. Builds auditable evidence logs.
    """
    subscriptions = []
    
    for merchant, group in df.groupby('merchant'):
        if len(group) < 2:
            continue  # Rule: Do not classify a transaction as recurring based on only one occurrence.
            
        group_sorted = group.sort_values('date').copy()
        dates = group_sorted['date'].tolist()
        amounts = group_sorted['amount'].tolist()
        descriptions = group_sorted['description'].tolist()
        category = group_sorted['category'].iloc[0]
        cancel_guide = group_sorted['cancel_guide'].iloc[0]
        
        # Calculate day intervals between consecutive transactions
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        mean_int = float(np.mean(intervals))
        std_int = float(np.std(intervals)) if len(intervals) > 1 else 0.0
        
        # Check cadence
        cadence = classify_cadence(mean_int, std_int)
        if cadence is None:
            continue  # Intervals do not conform to standard recurring billing cycles
            
        # Amount consistency check (subscriptions have fixed price or rare step-up price hike)
        unique_amounts = sorted(list(set(amounts)))
        amt_cv = (np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0
        has_autopay_kw = any(any(kw in str(d).upper() for kw in ['AUTOPAY', 'E-MANDATE', 'MANDATE', 'SUBSCRIPTION', 'MEMBERSHIP', 'PREMIUM']) for d in descriptions)
        
        # Guard against variable purchases (e.g. groceries or fuel) that happen to occur monthly
        if len(unique_amounts) > 2 and amt_cv > 0.15 and not has_autopay_kw:
            continue
            
        latest_amount = amounts[-1]
        earliest_amount = amounts[0]
        
        # Calculate normalized monthly cost
        if cadence == 'Weekly':
            monthly_cost = latest_amount * (52.0 / 12.0)
            annual_cost = latest_amount * 52.0
        elif cadence == 'Bi-Weekly':
            monthly_cost = latest_amount * (26.0 / 12.0)
            annual_cost = latest_amount * 26.0
        elif cadence == 'Monthly':
            monthly_cost = latest_amount
            annual_cost = latest_amount * 12.0
        elif cadence == 'Quarterly':
            monthly_cost = latest_amount / 3.0
            annual_cost = latest_amount * 4.0
        elif cadence == 'Semi-Annual':
            monthly_cost = latest_amount / 6.0
            annual_cost = latest_amount * 2.0
        elif cadence == 'Annual':
            monthly_cost = latest_amount / 12.0
            annual_cost = latest_amount
        else:
            monthly_cost = latest_amount
            annual_cost = latest_amount * 12.0
            
        # Project next billing date
        latest_date = dates[-1]
        est_days = round(mean_int)
        next_renewal = latest_date + timedelta(days=est_days)
        # Advance if next_renewal is in the past relative to latest statement date
        max_stmt_date = df['date'].max()
        while next_renewal <= max_stmt_date:
            next_renewal += timedelta(days=est_days)
            
        # Calculate interval regularity confidence (0 to 100%)
        regularity_score = max(60, min(100, int(100 - (std_int * 8))))
        
        # Build auditable evidence string
        date_first_str = dates[0].strftime('%Y-%m-%d')
        date_last_str = dates[-1].strftime('%Y-%m-%d')
        evidence = (
            f"{len(group)} transactions detected between {date_first_str} and {date_last_str}. "
            f"Mean interval: {mean_int:.1f} days (variance: ±{std_int:.1f}d). "
            f"Cadence classified as {cadence} with {regularity_score}% interval consistency."
        )
        
        subscriptions.append({
            'merchant': merchant,
            'category': category,
            'cadence': cadence,
            'charge_count': len(group),
            'latest_amount': latest_amount,
            'earliest_amount': earliest_amount,
            'monthly_cost': monthly_cost,
            'annual_cost': annual_cost,
            'mean_interval': mean_int,
            'std_interval': std_int,
            'regularity_score': regularity_score,
            'first_charged': dates[0],
            'latest_charged': dates[-1],
            'next_renewal': next_renewal,
            'evidence': evidence,
            'cancel_guide': cancel_guide,
            'raw_records': group_sorted
        })
        
    return sorted(subscriptions, key=lambda x: x['monthly_cost'], reverse=True)


def detect_price_increases(subscriptions: list):
    """
    Compares historical vs latest transaction amounts for each detected recurring merchant.
    Identifies step-up price increases, calculates percentage increase and extra annual cost.
    """
    price_hikes = []
    
    for sub in subscriptions:
        records = sub['raw_records'].sort_values('date')
        amounts = records['amount'].tolist()
        dates = records['date'].tolist()
        
        if len(amounts) >= 2:
            prev_amount = amounts[0]
            latest_amount = amounts[-1]
            
            # Detect genuine price increase (greater than ₹1 and > 2% to avoid rounding noise)
            if latest_amount > prev_amount and (latest_amount - prev_amount) >= 1.0:
                diff = latest_amount - prev_amount
                pct_increase = (diff / prev_amount) * 100.0
                
                # Find effective date when higher price first appeared
                hike_idx = next((i for i in range(1, len(amounts)) if amounts[i] > prev_amount), len(amounts)-1)
                hike_date = dates[hike_idx]
                
                # Calculate extra annual burden caused by the price increase
                cadence = sub['cadence']
                if cadence == 'Weekly':
                    extra_annual = diff * 52.0
                elif cadence == 'Monthly':
                    extra_annual = diff * 12.0
                elif cadence == 'Quarterly':
                    extra_annual = diff * 4.0
                elif cadence == 'Annual':
                    extra_annual = diff
                else:
                    extra_annual = diff * 12.0
                    
                price_hikes.append({
                    'merchant': sub['merchant'],
                    'category': sub['category'],
                    'cadence': sub['cadence'],
                    'prev_amount': prev_amount,
                    'latest_amount': latest_amount,
                    'increase': diff,
                    'pct_increase': pct_increase,
                    'effective_date': hike_date,
                    'extra_annual': extra_annual,
                    'total_charges_at_new_price': sum(1 for a in amounts if a >= latest_amount)
                })
                
    return sorted(price_hikes, key=lambda x: x['increase'], reverse=True)


def get_forgotten_subscription_flags(subscriptions: list, price_hikes: list):
    """
    Identifies repeated subscription payments that users should audit and review.
    Complies strictly with Hackathon requirements:
    'You have been charged repeatedly for this service. Check whether you still need this subscription.'
    """
    review_list = []
    hike_merchants = {h['merchant'] for h in price_hikes}
    
    # Categorize to detect service duplication (e.g., multiple video streaming services)
    cat_counts = {}
    for sub in subscriptions:
        cat_counts[sub['category']] = cat_counts.get(sub['category'], 0) + 1
        
    for sub in subscriptions:
        reasons = []
        
        # 1. Flag if charged repeatedly over multiple cycles (>= 3 billing occurrences)
        if sub['charge_count'] >= 3:
            reasons.append(f"Recurring auto-debit has been charged {sub['charge_count']} times without manual verification.")
            
        # 2. Flag if price increased silently
        if sub['merchant'] in hike_merchants:
            reasons.append("Recent price hike detected on this auto-debit mandate.")
            
        # 3. Flag category redundancy (e.g. multiple streaming services)
        if cat_counts.get(sub['category'], 0) >= 2 and 'Streaming' in sub['category']:
            reasons.append(f"Multiple active subscriptions detected in '{sub['category']}'.")
            
        # 4. Long running subscription (> 6 months)
        duration_days = (sub['latest_charged'] - sub['first_charged']).days
        if duration_days >= 180:
            reasons.append(f"Auto-debit active for {duration_days // 30} consecutive months.")
            
        if reasons:
            total_spent = sub['raw_records']['amount'].sum()
            review_list.append({
                'merchant': sub['merchant'],
                'category': sub['category'],
                'cadence': sub['cadence'],
                'charge_count': sub['charge_count'],
                'total_spent': total_spent,
                'monthly_cost': sub['monthly_cost'],
                'reasons': reasons,
                'audit_message': "You have been charged repeatedly for this service. Check whether you still need this subscription."
            })
            
    return review_list


# ==============================================================================
# 3. MAIN STREAMLIT UI APPLICATION
# ==============================================================================

def main():
    # Modern CSS Styling for White and Light-Purple Fintech Theme
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main Background */
        .stApp {
            background-color: #F8F9FE;
            color: #1E1B4B;
        }
        
        /* Metric Card Styling */
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #EDE9FE;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 16px -2px rgba(108, 92, 231, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px -4px rgba(108, 92, 231, 0.14);
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6B7280;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #1E1B4B;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #7C3AED;
            font-weight: 500;
            margin-top: 6px;
        }
        
        /* Alert Banners */
        .alert-card-danger {
            background: #FFF5F5;
            border-left: 5px solid #EF4444;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.08);
        }
        .alert-card-warning {
            background: #FFFBEB;
            border-left: 5px solid #F59E0B;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
        }
        .alert-card-info {
            background: #F5F3FF;
            border-left: 5px solid #6C5CE7;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(108, 92, 231, 0.08);
        }
        
        /* Custom Badges */
        .badge-purple {
            background-color: #EDE9FE;
            color: #4338CA;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        .badge-hike {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-green {
            background-color: #DCFCE7;
            color: #14532D;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        
        /* Header Container */
        .brand-header {
            background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
            border-radius: 18px;
            padding: 26px 32px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 8px 30px rgba(30, 27, 75, 0.18);
        }
        .brand-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-subtitle {
            color: #C7D2FE;
            font-size: 0.98rem;
            margin-top: 8px;
            line-height: 1.5;
        }
        
        /* Subscription Card in Grid */
        .sub-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }
        
        /* Streamlit Tab Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #EDE9FE;
            padding: 6px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            font-weight: 600;
            color: #4B5563;
            padding: 8px 16px;
            background-color: transparent;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #6C5CE7 !important;
            box-shadow: 0 2px 8px rgba(108, 92, 231, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:15px;">
            <span style="font-size:2rem;">🛡️</span>
            <div>
                <div style="font-size:1.3rem; font-weight:800; color:#1E1B4B; line-height:1.1;">SubSentry</div>
                <div style="font-size:0.75rem; color:#6C5CE7; font-weight:600;">FINTECH T4.1 PROTOTYPE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Smart Auto-Debit & Subscription Auditor**")
        st.caption("Detects hidden auto-debits, exposes price hikes, and audits recurring spending.")
        st.divider()
        
        st.subheader("📂 Ingest Bank Statement")
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'], help="Accepts CSV with date, description, and amount columns.")
        
        col_demo, col_down = st.columns(2)
        with col_demo:
            demo_clicked = st.button("⚡ Load Demo", help="Instantly load synthetic demo bank transactions", use_container_width=True)
        with col_down:
            if os.path.exists("sample_transactions.csv"):
                with open("sample_transactions.csv", "rb") as f:
                    st.download_button(
                        label="📥 CSV File",
                        data=f,
                        file_name="sample_transactions.csv",
                        mime="text/csv",
                        help="Download the sample transactions CSV for offline testing",
                        use_container_width=True
                    )
        
        st.divider()
        st.markdown("""
        <div style="background:#F1F0FF; padding:12px; border-radius:10px; font-size:0.78rem; color:#4338CA;">
            <strong>Hackathon Track:</strong> Fintech (T4)<br>
            <strong>Problem:</strong> T4.1 – The Auto Debits Nobody Can Find<br>
            <strong>Event:</strong> hackFront India
        </div>
        """, unsafe_allow_html=True)

    # State Management for Data Loading
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None

    if demo_clicked or (st.session_state.df_raw is None and uploaded_file is None and os.path.exists("sample_transactions.csv")):
        try:
            st.session_state.df_raw = pd.read_csv("sample_transactions.csv")
            st.session_state.data_source = "Demo Statement (sample_transactions.csv)"
        except Exception as e:
            st.error(f"Error loading demo file: {e}")

    if uploaded_file is not None:
        try:
            st.session_state.df_raw = pd.read_csv(uploaded_file)
            st.session_state.data_source = f"Uploaded File ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    # Header Branding Banner
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">
            <span>🛡️ SubSentry</span>
            <span class="badge-purple" style="background:#6C5CE7; color:#FFFFFF; font-size:0.8rem; vertical-align:middle;">hackFront India | Fintech T4.1</span>
        </div>
        <div class="brand-subtitle">
            Intelligent Auto-Debit & Subscription Auditor — Discover forgotten recurring debits, detect silent price increases, and manage your financial mandates.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.info("👋 Welcome to SubSentry! Upload your bank transaction CSV from the sidebar or click **⚡ Load Demo** to evaluate the application.")
        st.stop()

    # Data Cleaning & Validation
    try:
        df_clean, inv_dates, inv_amts = clean_and_validate_csv(st.session_state.df_raw)
    except Exception as e:
        st.error(f"⚠️ Validation Error: {e}")
        st.stop()

    # Run Algorithms
    subscriptions = detect_subscriptions(df_clean)
    price_hikes = detect_price_increases(subscriptions)
    review_flags = get_forgotten_subscription_flags(subscriptions, price_hikes)

    # Aggregated Metrics Calculations
    total_subs_count = len(subscriptions)
    monthly_sub_spend = sum(s['monthly_cost'] for s in subscriptions)
    annual_sub_spend = sum(s['annual_cost'] for s in subscriptions)
    total_hikes_count = len(price_hikes)
    annual_hike_drain = sum(h['extra_annual'] for h in price_hikes)
    total_transactions = len(df_clean)

    # Quick Info Bar
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption(f"📊 Statement Source: **{st.session_state.get('data_source', 'Custom Data')}** | Date Range: **{df_clean['date'].min().strftime('%d %b %Y')}** to **{df_clean['date'].max().strftime('%d %b %Y')}**")
    with c2:
        st.caption(f"Total Transactions: **{total_transactions}**")
    with c3:
        if inv_dates > 0 or inv_amts > 0:
            st.caption(f"Filtered Rows: **{inv_dates + inv_amts} invalid**")
        else:
            st.caption("Data Health: **100% Valid**")

    # Top KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Subscriptions</div>
            <div class="metric-value">{total_subs_count}</div>
            <div class="metric-sub">Auto-Debits Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Monthly Expenditure</div>
            <div class="metric-value">₹{monthly_sub_spend:,.2f}</div>
            <div class="metric-sub">Recurring monthly burn</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Annual Expenditure</div>
            <div class="metric-value">₹{annual_sub_spend:,.2f}</div>
            <div class="metric-sub">Projected 12-month drain</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        hike_color = "#EF4444" if total_hikes_count > 0 else "#10B981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Price Increases</div>
            <div class="metric-value" style="color:{hike_color};">{total_hikes_count}</div>
            <div class="metric-sub">+₹{annual_hike_drain:,.0f}/yr Extra Drain</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        review_color = "#F59E0B" if len(review_flags) > 0 else "#10B981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mandate Audits</div>
            <div class="metric-value" style="color:{review_color};">{len(review_flags)}</div>
            <div class="metric-sub">Review Recommended</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Prominent Price Increase Alert Banner (If Any Detected)
    if price_hikes:
        for hike in price_hikes:
            st.markdown(f"""
            <div class="alert-card-danger">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <span class="badge-hike">⚠️ PRICE INCREASE DETECTED</span>
                        <strong style="font-size:1.15rem; margin-left:8px; color:#991B1B;">{hike['merchant']}</strong>
                        <span style="color:#6B7280; font-size:0.9rem;">({hike['category']})</span>
                        <div style="margin-top:6px; font-size:0.95rem; color:#4B5563;">
                            Previous charge: <strong>₹{hike['prev_amount']:,.2f}</strong> &rarr; Latest charge: <strong>₹{hike['latest_amount']:,.2f}</strong>
                            <span style="color:#B91C1C; font-weight:700; margin-left:8px;">(+₹{hike['increase']:,.2f} / +{hike['pct_increase']:.2f}%)</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem; color:#6B7280;">Hike Effective Date</div>
                        <div style="font-size:1.05rem; font-weight:700; color:#1E1B4B;">{hike['effective_date'].strftime('%d %b %Y')}</div>
                        <div style="font-size:0.8rem; color:#DC2626; font-weight:600;">+₹{hike['extra_annual']:,.0f}/yr Annual Impact</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Multi-Tab Navigation Interface
    tab_analytics, tab_subs, tab_hikes, tab_forgotten, tab_cancel, tab_txns = st.tabs([
        "📊 Spending Analytics",
        "🔍 Detected Subscriptions",
        "⚠️ Price Increase Monitor",
        "🕵️ Forgotten Mandate Audit",
        "🛑 Cancellation Guidance",
        "📋 Statement Explorer"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SPENDING ANALYTICS
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.subheader("Subscription Spending Analytics & Renewal Horizon")
        
        if total_subs_count == 0:
            st.warning("No recurring subscriptions detected in the current statement.")
        else:
            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.markdown("##### 🍩 Monthly Spend Distribution by Merchant")
                df_sub_summary = pd.DataFrame([{
                    'Merchant': s['merchant'],
                    'Monthly Cost': s['monthly_cost'],
                    'Cadence': s['cadence'],
                    'Category': s['category']
                } for s in subscriptions])
                
                fig_donut = px.pie(
                    df_sub_summary,
                    values='Monthly Cost',
                    names='Merchant',
                    hole=0.55,
                    color_discrete_sequence=['#6C5CE7', '#805AD5', '#A29BFE', '#6366F1', '#8B5CF6', '#EC4899', '#3B82F6', '#10B981']
                )
                fig_donut.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Monthly Cost: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>"
                )
                fig_donut.update_layout(
                    showlegend=True,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=340
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col_chart2:
                st.markdown("##### 📊 Highest-Cost Subscriptions (Monthly Equivalent)")
                df_sorted_cost = df_sub_summary.sort_values('Monthly Cost', ascending=True)
                
                fig_bar = px.bar(
                    df_sorted_cost,
                    x='Monthly Cost',
                    y='Merchant',
                    orientation='h',
                    color='Monthly Cost',
                    color_continuous_scale=['#C7D2FE', '#6C5CE7', '#4338CA']
                )
                fig_bar.update_traces(
                    texttemplate="₹%{x:,.2f}",
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Monthly Equivalent: ₹%{x:,.2f}<extra></extra>"
                )
                fig_bar.update_layout(
                    showlegend=False,
                    xaxis_title="Monthly Equivalent Cost (₹)",
                    yaxis_title="",
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=10, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=340
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
            st.divider()
            
            st.markdown("##### 📈 Monthly Spend Trend: Subscriptions vs. Discretionary Transactions")
            recurring_merchants = {s['merchant'] for s in subscriptions}
            df_clean_trend = df_clean.copy()
            df_clean_trend['Spend Type'] = df_clean_trend['merchant'].apply(
                lambda m: 'Recurring Subscription' if m in recurring_merchants else 'Discretionary / One-Time'
            )
            df_clean_trend['YearMonth'] = df_clean_trend['date'].dt.to_period('M').astype(str)
            
            trend_grouped = df_clean_trend.groupby(['YearMonth', 'Spend Type'])['amount'].sum().reset_index()
            
            fig_trend = px.bar(
                trend_grouped,
                x='YearMonth',
                y='amount',
                color='Spend Type',
                barmode='stack',
                color_discrete_map={
                    'Recurring Subscription': '#6C5CE7',
                    'Discretionary / One-Time': '#CBD5E1'
                }
            )
            fig_trend.update_layout(
                xaxis_title="Billing Month",
                yaxis_title="Total Outflow (₹)",
                legend_title="",
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=320
            )
            fig_trend.update_traces(
                hovertemplate="<b>%{x}</b><br>%{data.name}: ₹%{y:,.2f}<extra></extra>"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.divider()
            
            st.markdown("##### 🗓️ Upcoming Renewal Schedule")
            today = df_clean['date'].max()
            renewal_list = []
            for s in subscriptions:
                days_left = (s['next_renewal'] - today).days
                renewal_list.append({
                    'Merchant': s['merchant'],
                    'Category': s['category'],
                    'Cadence': s['cadence'],
                    'Billing Amount': f"₹{s['latest_amount']:,.2f}",
                    'Next Projected Debit': s['next_renewal'].strftime('%d %b %Y'),
                    'Days Remaining': f"{days_left} days",
                    'days_int': days_left
                })
                
            df_renewal = pd.DataFrame(renewal_list).sort_values('days_int').drop(columns=['days_int'])
            st.dataframe(df_renewal, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: DETECTED SUBSCRIPTIONS & EVIDENCE
    # --------------------------------------------------------------------------
    with tab_subs:
        st.subheader("Audited Recurring Payment Mandates")
        st.caption("Transactions grouped by merchant, tested against cadence intervals, and verified with auditable transaction evidence.")
        
        for sub in subscriptions:
            with st.container():
                st.markdown(f"""
                <div class="sub-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
                        <div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:1.25rem; font-weight:700; color:#1E1B4B;">{sub['merchant']}</span>
                                <span class="badge-purple">{sub['cadence']}</span>
                                <span class="badge-green">{sub['category']}</span>
                            </div>
                            <div style="color:#6B7280; font-size:0.85rem; margin-top:4px;">
                                First billed: <strong>{sub['first_charged'].strftime('%d %b %Y')}</strong> | 
                                Latest debit: <strong>{sub['latest_charged'].strftime('%d %b %Y')}</strong> | 
                                Next Renewal: <strong style="color:#6C5CE7;">{sub['next_renewal'].strftime('%d %b %Y')}</strong>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:1.4rem; font-weight:800; color:#1E1B4B;">₹{sub['latest_amount']:,.2f}</div>
                            <div style="font-size:0.8rem; color:#6B7280;">₹{sub['monthly_cost']:,.2f}/month equiv.</div>
                        </div>
                    </div>
                    <div style="background:#F8F9FE; border-left:3px solid #6C5CE7; padding:10px 14px; border-radius:6px; margin-top:12px; font-size:0.84rem; color:#374151;">
                        <strong>🔍 Detection Evidence:</strong> {sub['evidence']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"View all {sub['charge_count']} charges for {sub['merchant']}"):
                    charges_view = sub['raw_records'][['date', 'description', 'amount']].copy()
                    charges_view['date'] = charges_view['date'].dt.strftime('%Y-%m-%d')
                    charges_view['amount'] = charges_view['amount'].apply(lambda a: f"₹{a:,.2f}")
                    st.dataframe(charges_view, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: PRICE INCREASE MONITOR
    # --------------------------------------------------------------------------
    with tab_hikes:
        st.subheader("Price Increase Detection & Historical Comparison")
        st.caption("SubSentry monitors recurring payment history to catch unnotified price increases and calculate your compounding financial loss.")
        
        if not price_hikes:
            st.success("✅ Good news! No price increases were detected across your active recurring subscriptions.")
        else:
            st.markdown(f"Found **{len(price_hikes)}** auto-debit price increases.")
            
            for hike in price_hikes:
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #EF4444;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.3rem; font-weight:700; color:#1E1B4B;">{hike['merchant']}</span>
                            <span class="badge-hike">+{hike['pct_increase']:.2f}% Hike</span>
                        </div>
                        <p style="color:#6B7280; font-size:0.85rem; margin-top:2px;">Category: {hike['category']} | Billing: {hike['cadence']}</p>
                        <hr style="margin:10px 0; border:none; border-top:1px solid #F3F4F6;">
                        <table style="width:100%; font-size:0.92rem; color:#374151;">
                            <tr>
                                <td>Previous Charge:</td>
                                <td style="text-align:right; font-weight:600;">₹{hike['prev_amount']:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Latest Charge:</td>
                                <td style="text-align:right; font-weight:700; color:#B91C1C;">₹{hike['latest_amount']:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Charge Increase:</td>
                                <td style="text-align:right; font-weight:700; color:#EF4444;">+₹{hike['increase']:,.2f}</td>
                            </tr>
                            <tr>
                                <td>Percentage Increase:</td>
                                <td style="text-align:right; font-weight:700; color:#EF4444;">+{hike['pct_increase']:.2f}%</td>
                            </tr>
                            <tr>
                                <td>Hike Inception Date:</td>
                                <td style="text-align:right; font-weight:600;">{hike['effective_date'].strftime('%d %b %Y')}</td>
                            </tr>
                            <tr>
                                <td>Extra Annual Outflow:</td>
                                <td style="text-align:right; font-weight:800; color:#DC2626;">+₹{hike['extra_annual']:,.2f} / yr</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_right:
                    sub_rec = next(s for s in subscriptions if s['merchant'] == hike['merchant'])
                    df_hist = sub_rec['raw_records'].copy()
                    
                    fig_hist = px.line(
                        df_hist,
                        x='date',
                        y='amount',
                        markers=True,
                        title=f"Price History: {hike['merchant']}",
                        line_shape='hv',
                        color_discrete_sequence=['#EF4444']
                    )
                    fig_hist.update_layout(
                        xaxis_title="Billing Date",
                        yaxis_title="Amount Charged (₹)",
                        margin=dict(t=35, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=260
                    )
                    fig_hist.update_traces(
                        hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Billed:</b> ₹%{y:,.2f}<extra></extra>"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                st.divider()

    # --------------------------------------------------------------------------
    # TAB 4: FORGOTTEN MANDATE AUDIT
    # --------------------------------------------------------------------------
    with tab_forgotten:
        st.subheader("Potential Forgotten Subscriptions & Mandate Review")
        st.caption("SubSentry highlights active recurring charges that have run continuously on auto-debit, undergone price changes, or overlap with duplicate services.")
        
        if not review_flags:
            st.success("No suspicious or redundant recurring mandates flagged for review.")
        else:
            for item in review_flags:
                st.markdown(f"""
                <div class="alert-card-warning">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
                        <div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:1.2rem; font-weight:700; color:#92400E;">{item['merchant']}</span>
                                <span class="badge-purple">{item['cadence']}</span>
                                <span class="badge-green">{item['category']}</span>
                            </div>
                            <div style="margin-top:6px; font-size:0.95rem; font-weight:600; color:#B45309;">
                                ⚠️ {item['audit_message']}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.85rem; color:#6B7280;">Cumulative Total Paid</div>
                            <div style="font-size:1.3rem; font-weight:800; color:#1E1B4B;">₹{item['total_spent']:,.2f}</div>
                            <div style="font-size:0.8rem; color:#6B7280;">Across {item['charge_count']} debit cycles</div>
                        </div>
                    </div>
                    <div style="margin-top:10px; padding-top:8px; border-top:1px dashed #FDE68A;">
                        <strong style="font-size:0.82rem; color:#78350F;">Audit Signals:</strong>
                        <ul style="margin:4px 0 0 16px; font-size:0.85rem; color:#4B5563;">
                            {''.join(f"<li>{r}</li>" for r in item['reasons'])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.info("💡 **SubSentry Tip**: Check your usage frequency for the flagged services. If you haven't opened an app or consumed content in the past 30 days, consider pausing or cancelling the mandate to save money.")

    # --------------------------------------------------------------------------
    # TAB 5: CANCELLATION GUIDANCE
    # --------------------------------------------------------------------------
    with tab_cancel:
        st.subheader("Actionable Cancellation & Mandate Revocation Playbook")
        st.caption("Step-by-step instructions to cancel recurring subscriptions and revoke underlying UPI AutoPay or Credit Card e-mandates.")
        
        st.markdown("""
        <div class="alert-card-info">
            <strong>🔒 Mandate Protection Notice:</strong> In India, cancelling an account inside an app doesn't always delete the bank e-mandate. 
            Always verify both: <strong>(1) In-App Cancellation</strong> AND <strong>(2) UPI AutoPay / Netbanking e-mandate revocation</strong> at least 24 hours prior to renewal.
        </div>
        """, unsafe_allow_html=True)
        
        for sub in subscriptions:
            with st.expander(f"🛑 How to cancel {sub['merchant']} (Saving ₹{sub['monthly_cost']:,.2f}/mo)"):
                c_left, c_right = st.columns([1, 1])
                
                with c_left:
                    st.markdown("#### 📱 Step 1: Direct Service Cancellation")
                    st.write(sub['cancel_guide'])
                    st.markdown(f"""
                    - **Billing Cadence:** {sub['cadence']}
                    - **Current Amount:** ₹{sub['latest_amount']:,.2f}
                    - **Next Projected Billing Date:** {sub['next_renewal'].strftime('%d %b %Y')}
                    """)
                    
                with c_right:
                    st.markdown("#### 💳 Step 2: Revoke Bank Auto-Debit Mandate")
                    st.markdown("""
                    Choose your payment method:
                    - **UPI AutoPay (PhonePe / Google Pay / Paytm):**
                      1. Open your UPI app and tap your **Profile photo**.
                      2. Select **AutoPay / Automatic Payments / Mandates**.
                      3. Find the mandate for this merchant.
                      4. Tap **Revoke Mandate / Cancel AutoPay** and enter your **UPI PIN**.
                    - **Credit/Debit Card (RBI SI-Hub / Bank Netbanking):**
                      1. Log into your bank netbanking (HDFC, ICICI, SBI, Axis, etc.).
                      2. Go to **Cards > Standing Instructions / e-Mandates Hub**.
                      3. Locate the merchant and select **Stop / Cancel Instruction**.
                    - **Google Play / App Store:**
                      - Android: Play Store > Profile > Payments & Subscriptions > Subscriptions > Cancel.
                      - iOS: Settings > Apple ID > Subscriptions > Select Service > Cancel.
                    """)
                    
                st.warning("⚠️ **Reminder:** Retain the cancellation confirmation email and check your bank statement in the next billing cycle to ensure no further debits occur.")

    # --------------------------------------------------------------------------
    # TAB 6: STATEMENT EXPLORER
    # --------------------------------------------------------------------------
    with tab_txns:
        st.subheader("Bank Transaction Statement Explorer")
        st.caption("Explore all parsed transactions, filter by subscription classification, search by description, or export audit reports.")
        
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search Description / Merchant", placeholder="e.g. Netflix, Spotify, Fuel, Swiggy...")
        with col_f2:
            class_filter = st.selectbox("Classification", ["All Transactions", "Recurring Subscriptions Only", "Discretionary / One-Time"])
        with col_f3:
            sort_order = st.selectbox("Sort By", ["Date (Newest First)", "Date (Oldest First)", "Amount (High to Low)"])
            
        recurring_merchants_set = {s['merchant'] for s in subscriptions}
        df_display = df_clean.copy()
        df_display['Is Recurring'] = df_display['merchant'].apply(lambda m: "✅ Recurring" if m in recurring_merchants_set else "One-Time")
        
        if search_query:
            df_display = df_display[
                df_display['description'].str.contains(search_query, case=False, na=False) |
                df_display['merchant'].str.contains(search_query, case=False, na=False)
            ]
            
        if class_filter == "Recurring Subscriptions Only":
            df_display = df_display[df_display['Is Recurring'] == "✅ Recurring"]
        elif class_filter == "Discretionary / One-Time":
            df_display = df_display[df_display['Is Recurring'] == "One-Time"]
            
        if sort_order == "Date (Newest First)":
            df_display = df_display.sort_values('date', ascending=False)
        elif sort_order == "Date (Oldest First)":
            df_display = df_display.sort_values('date', ascending=True)
        elif sort_order == "Amount (High to Low)":
            df_display = df_display.sort_values('amount', ascending=False)
            
        st.markdown(f"Displaying **{len(df_display)}** of **{len(df_clean)}** transactions:")
        
        df_table = df_display[['date', 'description', 'merchant', 'category', 'amount', 'Is Recurring']].copy()
        df_table['date'] = df_table['date'].dt.strftime('%Y-%m-%d')
        df_table['amount'] = df_table['amount'].apply(lambda a: f"₹{a:,.2f}")
        
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        st.divider()
        sub_export_df = pd.DataFrame([{
            'Merchant': s['merchant'],
            'Category': s['category'],
            'Cadence': s['cadence'],
            'Charge Amount': s['latest_amount'],
            'Monthly Equivalent (INR)': round(s['monthly_cost'], 2),
            'Annual Projected (INR)': round(s['annual_cost'], 2),
            'First Charged Date': s['first_charged'].strftime('%Y-%m-%d'),
            'Latest Charged Date': s['latest_charged'].strftime('%Y-%m-%d'),
            'Next Predicted Renewal': s['next_renewal'].strftime('%Y-%m-%d'),
            'Total Occurrences': s['charge_count'],
            'Evidence': s['evidence']
        } for s in subscriptions])
        
        csv_bytes = sub_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Detected Subscriptions Report (CSV)",
            data=csv_bytes,
            file_name="subsentry_detected_subscriptions_report.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:0.85rem; padding:10px 0;">
        <strong>SubSentry</strong> • Built for <strong>hackFront India</strong> • Fintech Problem Statement T4.1 • <em>The Auto Debits Nobody Can Find</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
