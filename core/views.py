
import os
import hashlib
import time
import pandas as pd
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.conf import settings
from django.db import transaction
from django.contrib.admin.models import LogEntry, ADDITION, CHANGE, DELETION
from django.views.decorators.csrf import csrf_protect
from django.contrib.contenttypes.models import ContentType
from .models import User, Upload, MedicineData, ModelTrainingInfo, Prediction, Feedback


import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from django.core.paginator import Paginator
from django.db.models import Q
from django.db.models import Count, Avg, Max, Min, Sum, F, FloatField
from django.db.models.functions import TruncMonth, ExtractMonth, ExtractYear



# Create your views here.

def get_file_hash(file):
    """Calculate MD5 hash of file content to detect duplicates"""
    hasher = hashlib.md5()
    for chunk in file.chunks():
        hasher.update(chunk)
    return hasher.hexdigest()

def index(request):
    if request.user.is_authenticated:
        # User has valid session, proceed to view
        return redirect('dashboard')
    else:
        return render(request, 'index.html')
    

def loginView(request):
    if request.method == 'POST':
        username = request.POST['username'].strip()
        password = request.POST['password'].strip()
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, 'You are successfully logged-in.')
            next_url = request.GET.get('next', 'dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('index')
        
    return redirect('index')


def registerView(request):
    if request.method == 'POST':
        # Get form data
        first_name = request.POST.get('firstName')
        last_name = request.POST.get('lastName')
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Basic validation
        if not all([first_name, last_name, username, email, password]):
            messages.error(request, 'All fields are required.')
            return redirect('index')

        # Check if username exists
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('index')

        # Check if email exists
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists.')
            return redirect('index')

        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )
            
            # Set role based on user ID
            if user.id == 1:
                user.role = 'admin'
                user.is_staff = True
                user.is_superuser = True
            else:
                user.role = 'pharmacist'
            
            user.save()
            
            # Log user in
            login(request, user)
            messages.success(request, f'Account created successfully as {user.get_role_display()}!')
            return redirect('dashboard')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return redirect('index')

    # If GET request, redirect to home
    return redirect('index')

def logoutView(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('index')



@login_required
def dashboardView(request):
    """Display dashboard with statistics and charts"""
    
    # Get user's data
    user = request.user
    
    # 1. Count datasets (uploads)
    datasets_count = MedicineData.objects.filter(upload__user=user).count()
    
    # 2. Count trained models (only completed ones)
    trained_models_count = ModelTrainingInfo.objects.filter(
        user=user,
        status='completed'
    ).count()
    
    # 3. Count predictions
    predictions_count = Prediction.objects.filter(user=user).count()
    
    # 4. Get latest model accuracy
    latest_model = ModelTrainingInfo.objects.filter(
        user=user,
        status='completed'
    ).order_by('-trained_at').first()
    
    accuracy = latest_model.accuracy if latest_model else 0
    
    # 5. Get all predictions for charts
    predictions = Prediction.objects.filter(
        user=user
    ).select_related('model').values(
        'medicine_name',
        'brand_name',
        'category',
        'total_sales',
        'predicted_demand',
        'confidence',
        'date_from',
        'date_to',
        'created_at'
    ).order_by('-created_at')
    
    # 6. Get unique categories, brands, and date ranges
    categories = list(
        predictions.values_list('category', flat=True).distinct().order_by('category')
    )
    brands = list(
        predictions.values_list('brand_name', flat=True).distinct().order_by('brand_name')
    )
    medicine_names = list(
        predictions.values_list('medicine_name', flat=True).distinct().order_by('medicine_name')
    )
    
    # Create unique date ranges with proper formatting and sorting
    date_range_objects = []
    seen_ranges = set()
    for pred in predictions:
        date_from = pred['date_from']
        date_to = pred['date_to']
        # Use ISO format for value (filtering)
        range_value = f"{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}"
        if range_value not in seen_ranges:
            seen_ranges.add(range_value)
            date_range_objects.append({
                'value': range_value,
                'date_from': date_from  # Keep for sorting
            })
    
    # Sort by date_from (latest/most recent first)
    date_range_objects.sort(key=lambda x: x['date_from'], reverse=True)
    
    # Extract just the values for the template
    date_ranges = [dr['value'] for dr in date_range_objects]
    
    # 7. Prepare prediction data for JavaScript
    prediction_data = []
    for pred in predictions:
        # Use ISO format for period (for filtering consistency)
        period_value = f"{pred['date_from'].strftime('%Y-%m-%d')} to {pred['date_to'].strftime('%Y-%m-%d')}"
        prediction_data.append({
            'name': pred['medicine_name'] or 'Unknown',
            'category': pred['category'] or 'Unknown',
            'brand': pred['brand_name'] or 'Unknown',
            'demand': float(pred['total_sales'] or 0),
            'confidence': float(pred['confidence'] or 0),
            'date': pred['created_at'].strftime('%Y-%m-%d'),
            'period': period_value,
            'predicted_level': pred['predicted_demand'] or 'Unknown'
        })
    
    # 8. Calculate demand distribution
    high_demand_count = predictions.filter(predicted_demand='High').count()
    medium_demand_count = predictions.filter(predicted_demand='Medium').count()
    low_demand_count = predictions.filter(predicted_demand='Low').count()
    
    total_predictions = high_demand_count + medium_demand_count + low_demand_count
    
    # Calculate percentages
    high_demand_percent = round((high_demand_count / total_predictions * 100), 1) if total_predictions > 0 else 0
    medium_demand_percent = round((medium_demand_count / total_predictions * 100), 1) if total_predictions > 0 else 0
    low_demand_percent = round((low_demand_count / total_predictions * 100), 1) if total_predictions > 0 else 0
    
    context = {
        # Stats cards
        'datasets_count': datasets_count,
        'trained_models_count': trained_models_count,
        'predictions_count': predictions_count,
        'accuracy': round(accuracy, 2),
        
        # Filter options
        'categories': categories,
        'brands': brands,
        'medicine_names': medicine_names,
        'date_ranges': date_ranges,
        
        # Prediction data for charts (as JSON)
        'prediction_data': prediction_data,
        
        # Demand distribution
        'high_demand_count': high_demand_count,
        'medium_demand_count': medium_demand_count,
        'low_demand_count': low_demand_count,
        'high_demand_percent': high_demand_percent,
        'medium_demand_percent': medium_demand_percent,
        'low_demand_percent': low_demand_percent,
    }
    
    return render(request, 'dashboard.html', context)


# upload views
@login_required
def uploadView(request):
    return render(request, 'upload.html')


@login_required
def uploadView(request):
    """Display the upload page with recent uploads"""
    recent_uploads = Upload.objects.filter(user=request.user).select_related('user').order_by('-uploaded_at')[:10]
    
    # Add record count for each upload
    for upload in recent_uploads:
        upload.record_count = upload.medicine_records.count()
    
    context = {
        'username': request.user.username,
        'recent_uploads': recent_uploads,
    }
    return render(request, 'upload.html', context)


@login_required
def uploadCsv(request):
    """Handle CSV file upload with incremental data update support"""
    if request.method != 'POST':
        return redirect('uploadPage')

    file = request.FILES.get('csvUpload')
    if not file:
        messages.error(request, 'No file selected.')
        return redirect('uploadPage')

    if not file.name.lower().endswith('.csv'):
        messages.error(request, 'Only CSV files allowed.')
        return redirect('uploadPage')

    filepath = None

    try:
        # User-specific upload directory
        user_upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', str(request.user.id))
        os.makedirs(user_upload_dir, exist_ok=True)

        # Save file temporarily to read
        filepath = os.path.join(user_upload_dir, file.name)

        with open(filepath, 'wb+') as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        # Read CSV with utf-8-sig to handle BOM
        df = pd.read_csv(
            filepath,
            dtype=str,
            encoding='utf-8-sig',
            sep=',',
            on_bad_lines='skip',
            low_memory=False
        )

        # Strip column names and check required columns
        df.columns = df.columns.str.strip().str.lower()
        required_cols = {'date', 'medicine_name', 'category', 'brand_name', 'usage', 'qty'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            os.remove(filepath)
            messages.error(request, f'Missing required columns: {", ".join(missing_cols)}')
            return redirect('uploadPage')

        df = df[list(required_cols)]

        # Clean string fields
        for col in ['medicine_name', 'category', 'brand_name', 'usage']:
            df[col] = df[col].astype(str).str.strip().replace({'nan': '', 'None': '', 'NaN': ''})

        # Clean quantity
        df['qty'] = df['qty'].astype(str).str.replace(',', '').str.strip()
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce')

        # Clean date
        df['date'] = df['date'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)

        # Keep rows with valid critical fields
        valid_rows = df.dropna(subset=['date', 'medicine_name', 'brand_name', 'qty'])
        valid_rows = valid_rows[valid_rows['qty'] >= 0]

        total_rows = len(df)
        valid_count = len(valid_rows)
        skipped_rows = total_rows - valid_count

        if valid_count == 0:
            os.remove(filepath)
            messages.error(request, 'No valid rows to save.')
            return redirect('uploadPage')

        # ==================== CHECK FOR DUPLICATES ====================
        
        # Create unique key for each row (date + medicine_name + brand_name + category + qty)
        valid_rows = valid_rows.copy()
        valid_rows['record_key'] = (
            valid_rows['date'].dt.strftime('%Y-%m-%d') + '|' +
            valid_rows['medicine_name'].str.lower() + '|' +
            valid_rows['brand_name'].str.lower() + '|' +
            valid_rows['category'].str.lower() + '|' +
            valid_rows['qty'].astype(str)
        )

        # Get all existing records for this user
        existing_data = MedicineData.objects.filter(
            upload__user=request.user
        ).values('date', 'medicine_name', 'brand_name', 'category', 'qty')

        # Create keys for existing records
        existing_keys = set()
        for record in existing_data:
            key = (
                record['date'].strftime('%Y-%m-%d') + '|' +
                record['medicine_name'].lower() + '|' +
                record['brand_name'].lower() + '|' +
                record['category'].lower() + '|' +
                str(int(record['qty']) if record['qty'] == int(record['qty']) else record['qty'])
            )
            existing_keys.add(key)

        # Filter out duplicate rows
        new_rows = valid_rows[~valid_rows['record_key'].isin(existing_keys)]
        duplicate_count = valid_count - len(new_rows)

        # ==================== HANDLE DIFFERENT SCENARIOS ====================

        if len(new_rows) == 0:
            # All records already exist
            os.remove(filepath)
            messages.warning(
                request, 
                f'No new records to add. All {valid_count} records already exist in the database.'
            )
            return redirect('uploadPage')

        # Check if filename already exists
        existing_upload = Upload.objects.filter(user=request.user, filename=file.name).first()

        with transaction.atomic():
            if existing_upload:
                # Same filename - update existing upload
                upload_obj = existing_upload
                action_type = 'updated'
            else:
                # New filename - create new upload record
                upload_obj = Upload.objects.create(
                    user=request.user,
                    filename=file.name
                )
                action_type = 'created'

            # Save only new records
            batch = []
            chunk_size = 5000
            saved_count = 0

            for _, row in new_rows.iterrows():
                batch.append(
                    MedicineData(
                        upload=upload_obj,
                        date=row['date'].date(),
                        medicine_name=row['medicine_name'],
                        category=row['category'],
                        brand_name=row['brand_name'],
                        usage=row['usage'],
                        qty=row['qty']
                    )
                )

                if len(batch) >= chunk_size:
                    MedicineData.objects.bulk_create(batch)
                    saved_count += len(batch)
                    batch = []

            if batch:
                MedicineData.objects.bulk_create(batch)
                saved_count += len(batch)

            # Log the action
            LogEntry.objects.create(
                user_id=request.user.id,
                content_type_id=ContentType.objects.get_for_model(Upload).pk,
                object_id=str(upload_obj.id),
                object_repr=file.name,
                action_flag=ADDITION if action_type == 'created' else CHANGE,
                change_message=f'{saved_count} new records added, {duplicate_count} duplicates skipped, {skipped_rows} invalid rows skipped'
            )

        # Build success message
        if duplicate_count > 0:
            messages.success(
                request, 
                f'{saved_count} new records added. {duplicate_count} duplicate records skipped.'
            )
        else:
            messages.success(request, f'{saved_count} records uploaded successfully.')

        return redirect('uploadPage')

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        import traceback
        print(traceback.format_exc())
        messages.error(request, f'Upload error: {str(e)}')
        return redirect('uploadPage')

# end upload views

# training views

@login_required
def trainView(request):
    return render(request, 'train.html')

@login_required
def trainModel(request):
    """Handle model training with optimizations"""
    if request.method != 'POST':
        return redirect('trainPage')

    try:
        start_time = time.time()

        # Optimized query - select only needed fields
        qs = MedicineData.objects.filter(
            upload__user=request.user
        ).values(
            'date',
            'medicine_name',
            'brand_name',
            'category',
            'usage',
            'qty'
        )

        if not qs.exists():
            messages.warning(request, 'No data found for your account. Please upload data first.')
            return redirect('trainPage')

        # Convert to DataFrame efficiently
        df = pd.DataFrame.from_records(qs)
        initial_count = len(df)
        
        # Drop nulls
        df.dropna(inplace=True)
        
        if df.empty:
            messages.error(request, 'No valid data after cleaning.')
            return redirect('trainPage')

        # Lowercase columns
        df.columns = df.columns.str.lower()
        
        # Parse dates efficiently
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)

        # Monthly aggregation
        df['month'] = df['date'].dt.to_period('M')

        monthly = (
            df.groupby(
                ['medicine_name', 'brand_name', 'category', 'usage', 'month'],
                as_index=False
            )
            .agg(total_qty=('qty', 'sum'))
        )

        # Demand classification using quantiles (vectorized - faster)
        q_high = monthly['total_qty'].quantile(0.66)
        q_low = monthly['total_qty'].quantile(0.33)

        monthly['demand_class'] = pd.cut(
            monthly['total_qty'],
            bins=[-np.inf, q_low, q_high, np.inf],
            labels=['Low', 'Medium', 'High']
        )

        # Encoding target
        target_enc = LabelEncoder()
        monthly['demand_class_enc'] = target_enc.fit_transform(monthly['demand_class'])

        # Target encoding for brand (faster with dict)
        brand_target_map = monthly.groupby('brand_name')['demand_class_enc'].mean().to_dict()
        monthly['brand_target_enc'] = monthly['brand_name'].map(brand_target_map)

        # Time features
        monthly['year'] = monthly['month'].dt.year
        monthly['quarter'] = monthly['month'].dt.quarter
        monthly['month_num'] = monthly['month'].dt.month
        monthly['log_total_qty'] = np.log1p(monthly['total_qty'])

        # Label encoding
        name_enc = LabelEncoder()
        brand_enc = LabelEncoder()
        cat_enc = LabelEncoder()
        usage_enc = LabelEncoder()

        monthly['medicine_name_enc'] = name_enc.fit_transform(monthly['medicine_name'].astype(str))
        monthly['brand_name_enc'] = brand_enc.fit_transform(monthly['brand_name'].astype(str))
        monthly['category_enc'] = cat_enc.fit_transform(monthly['category'].astype(str))
        monthly['usage_enc'] = usage_enc.fit_transform(monthly['usage'].astype(str))

        # Feature matrix
        feature_cols = [
            'medicine_name_enc',
            'category_enc',
            'usage_enc',
            'month_num',
            'log_total_qty',
            'year',
            'quarter',
            'brand_target_enc'
        ]
        X = monthly[feature_cols].copy()
        y = monthly['demand_class_enc']

        # Scale numeric features
        scaler = StandardScaler()
        scale_cols = ['month_num', 'log_total_qty', 'year', 'quarter', 'brand_target_enc']
        X.loc[:, scale_cols] = scaler.fit_transform(X[scale_cols])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Optimized XGBoost parameters for MAXIMUM speed
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': 0.15,  # Increased for faster convergence
            'max_depth': 4,  # Reduced for speed
            'n_estimators': 150,  # Reduced for speed
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.3,  # Reduced for speed
            'min_child_weight': 3,
            'reg_alpha': 0.1,  # Reduced for speed
            'reg_lambda': 0.5,  # Reduced for speed
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'tree_method': 'hist',  # Fastest tree method
            'max_bin': 128,  # Reduced for speed (default is 256)
        }

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate inference time
        inference_start = time.time()
        _ = model.predict(X_test[:10])  # Test on 10 samples
        inference_time = round((time.time() - inference_start) * 100, 2)  # in ms

        # Metrics
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        precision = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        recall = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)

        training_time = round(time.time() - start_time, 2)
        training_minutes = round(training_time / 60, 2)

        # Save models in user-specific directory
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(request.user.id))
        os.makedirs(model_dir, exist_ok=True)

        # Save all model artifacts
        joblib.dump(model, f'{model_dir}/xgb_model.pkl')
        joblib.dump(name_enc, f'{model_dir}/name_encoder.pkl')
        joblib.dump(brand_enc, f'{model_dir}/brand_encoder.pkl')
        joblib.dump(cat_enc, f'{model_dir}/category_encoder.pkl')
        joblib.dump(usage_enc, f'{model_dir}/usage_encoder.pkl')
        joblib.dump(target_enc, f'{model_dir}/target_encoder.pkl')
        joblib.dump(scaler, f'{model_dir}/scaler.pkl')
        joblib.dump(brand_target_map, f'{model_dir}/brand_target_map.pkl')

        # Save training info to database
        training_info = ModelTrainingInfo.objects.create(
            user=request.user,
            model_type='XGBoost Classifier',
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            min_child_weight=params['min_child_weight'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            accuracy=acc,
            training_time=training_time,
            data_points=initial_count,
            status='completed'
        )

        # Create audit log entry
        LogEntry.objects.create(
            user_id=request.user.id,
            content_type_id=ContentType.objects.get_for_model(ModelTrainingInfo).pk,
            object_id=training_info.id,
            object_repr=f'XGBoost Model - {acc}% accuracy',
            action_flag=ADDITION,
            change_message=f'Model trained: {initial_count} records, {acc}% accuracy, {training_time}s training time'
        )

        messages.success(request, f'Model trained successfully! Accuracy: {acc}%')

        # Return results
        return render(
            request,
            'train.html',
            {
                'show_results': True,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_time': training_time,
                'training_minutes': training_minutes,
                'record_count': initial_count,
                'processed_count': len(monthly),
                'model_version': 'v2.1',
                'status': 'Completed',
                'training_id': training_info.id,
                'inference_time': inference_time,
            }
        )

    except Exception as e:
        messages.error(request, f'Training error: {str(e)}')
        return redirect('trainPage')




# end training views








@login_required
def predictView(request):
    return render(request, 'predict.html')

@login_required
def makePrediction(request):
    """Handle prediction with optimizations"""
    if request.method != 'POST':
        return redirect('predictPage')

    try:
        # Get date inputs
        date_from_str = request.POST.get('date_from')
        date_to_str = request.POST.get('date_to')

        if not date_from_str or not date_to_str:
            messages.warning(request, 'Please select both start and end dates.')
            return redirect('predictPage')

        date_from = pd.to_datetime(date_from_str)
        date_to = pd.to_datetime(date_to_str)

        # Validate date range
        if date_from > date_to:
            messages.error(request, 'Start date must be before end date.')
            return redirect('predictPage')

        # ==================== CHECK FOR EXISTING PREDICTIONS ====================
        
        # Try multiple formats for date comparison
        from django.db.models import Q
        
        date_from_date = date_from.date()
        date_to_date = date_to.date()
        
        # Build query to check for existing predictions with same date range
        existing_query = Q(user=request.user) & (
            # Try as date objects
            (Q(date_from=date_from_date) & Q(date_to=date_to_date)) |
            # Try as strings (YYYY-MM-DD format)
            (Q(date_from=str(date_from_date)) & Q(date_to=str(date_to_date))) |
            # Try as original string format
            (Q(date_from=date_from_str) & Q(date_to=date_to_str))
        )
        
        existing_predictions = Prediction.objects.filter(existing_query)
        
        if existing_predictions.exists():
            count = existing_predictions.count()
            messages.warning(
                request, 
                f'Predictions for {date_from_str} to {date_to_str} already exist ({count} predictions). '
                f'Please choose a different date range or delete existing predictions first.'
            )
            return redirect('predictPage')

        # ==================== CONTINUE WITH PREDICTION ====================

        prediction_month = date_from.month
        prediction_year = date_from.year
        prediction_quarter = date_from.quarter

        # Load user-specific models
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models', str(request.user.id))
        
        if not os.path.exists(model_dir):
            messages.error(request, 'No trained model found. Please train a model first.')
            return redirect('predictPage')

        try:
            model = joblib.load(f'{model_dir}/xgb_model.pkl')
            name_enc = joblib.load(f'{model_dir}/name_encoder.pkl')
            brand_enc = joblib.load(f'{model_dir}/brand_encoder.pkl')
            cat_enc = joblib.load(f'{model_dir}/category_encoder.pkl')
            usage_enc = joblib.load(f'{model_dir}/usage_encoder.pkl')
            target_enc = joblib.load(f'{model_dir}/target_encoder.pkl')
            scaler = joblib.load(f'{model_dir}/scaler.pkl')
            brand_target_map = joblib.load(f'{model_dir}/brand_target_map.pkl')
        except FileNotFoundError as e:
            messages.error(request, 'Model files not found. Please retrain your model.')
            return redirect('predictPage')

        # Load user's medicine data
        qs = MedicineData.objects.filter(
            upload__user=request.user
        ).values(
            'date',
            'medicine_name',
            'brand_name',
            'category',
            'usage',
            'qty'
        )

        if not qs.exists():
            messages.warning(request, 'No data found. Please upload data first.')
            return redirect('predictPage')

        df = pd.DataFrame.from_records(qs)
        df['date'] = pd.to_datetime(df['date'])

        # Calculate day of year range for seasonal filtering
        start_doy = date_from.timetuple().tm_yday
        end_doy = date_to.timetuple().tm_yday

        def in_range(d):
            if d.year >= prediction_year:
                return False
            doy = d.timetuple().tm_yday
            if start_doy <= end_doy:
                return start_doy <= doy <= end_doy
            return doy >= start_doy or doy <= end_doy

        # Filter historical data for the same seasonal period
        hist_df = df[df['date'].apply(in_range)].copy()

        if hist_df.empty:
            messages.warning(
                request, 
                f'No historical data found for the period {date_from_str} to {date_to_str}.'
            )
            return redirect('predictPage')

        hist_df['year'] = hist_df['date'].dt.year

        # Aggregate by brand and year
        agg = hist_df.groupby(['brand_name', 'year'], as_index=False).agg(
            period_qty=('qty', 'sum')
        )

        # Calculate average quantity per brand
        avg_qty = agg.groupby('brand_name', as_index=False).agg(
            avg_period_qty=('period_qty', 'mean')
        )

        # Get latest info for each brand
        latest_info = df.sort_values('date', ascending=False).drop_duplicates('brand_name')[
            ['brand_name', 'medicine_name', 'category', 'usage']
        ]

        avg_qty = avg_qty.merge(latest_info, on='brand_name', how='left')

        # Default target encoding
        default_target = np.mean(list(brand_target_map.values()))

        # Build prediction DataFrame
        rows = []
        for _, r in avg_qty.iterrows():
            rows.append({
                'medicine_name': r['medicine_name'],
                'brand_name': r['brand_name'],
                'category': r['category'],
                'usage': r['usage'],
                'month': prediction_month,
                'year': prediction_year,
                'quarter': prediction_quarter,
                'brand_target_enc': brand_target_map.get(r['brand_name'], default_target),
                'total_qty': r['avg_period_qty']
            })

        pred_df = pd.DataFrame(rows)

        # Safe encoding function
        def encode_safe(val, enc):
            try:
                return enc.transform([str(val)])[0]
            except:
                return -1

        # Encode features
        pred_df['medicine_name_enc'] = pred_df['medicine_name'].apply(
            lambda x: encode_safe(x, name_enc)
        )
        pred_df['brand_name_enc'] = pred_df['brand_name'].apply(
            lambda x: encode_safe(x, brand_enc)
        )
        pred_df['category_enc'] = pred_df['category'].apply(
            lambda x: encode_safe(x, cat_enc)
        )
        pred_df['usage_enc'] = pred_df['usage'].apply(
            lambda x: encode_safe(x, usage_enc)
        )

        # Filter out rows with encoding failures
        pred_df = pred_df[
            (pred_df['medicine_name_enc'] != -1) &
            (pred_df['category_enc'] != -1)
        ]

        if pred_df.empty:
            messages.warning(request, 'No valid medicines found for prediction.')
            return redirect('predictPage')

        # Create features
        pred_df['month_num'] = pred_df['month']
        pred_df['log_total_qty'] = np.log1p(pred_df['total_qty'])

        # Feature matrix matching training
        X = pred_df[
            [
                'medicine_name_enc',
                'category_enc',
                'usage_enc',
                'month_num',
                'log_total_qty',
                'year',
                'quarter',
                'brand_target_enc'
            ]
        ].astype(float)

        # Scale features
        scale_cols = ['month_num', 'log_total_qty', 'year', 'quarter', 'brand_target_enc']
        X.loc[:, scale_cols] = scaler.transform(X[scale_cols])

        # Make predictions
        preds = model.predict(X)
        probs = model.predict_proba(X).max(axis=1)

        pred_df['predicted_demand'] = target_enc.inverse_transform(preds)
        pred_df['confidence'] = (probs * 100).round(2)

        # Get latest model info
        model_info = ModelTrainingInfo.objects.filter(
            user=request.user,
            status='completed'
        ).latest('trained_at')

        # Save predictions to database (use date objects)
        prediction_objects = []
        for _, r in pred_df.iterrows():
            prediction_objects.append(
                Prediction(
                    model=model_info,
                    user=request.user,
                    medicine_name=r['medicine_name'],
                    brand_name=r['brand_name'],
                    category=r['category'],
                    total_sales=float(r['total_qty']),
                    date_from=date_from_date,
                    date_to=date_to_date,
                    predicted_demand=r['predicted_demand'],
                    confidence=float(r['confidence'])
                )
            )

        # Bulk create predictions
        Prediction.objects.bulk_create(prediction_objects)

        # Create audit log entry
        LogEntry.objects.create(
            user_id=request.user.id,
            content_type_id=ContentType.objects.get_for_model(Prediction).pk,
            object_id=prediction_objects[0].id if prediction_objects else None,
            object_repr=f'Predictions: {date_from_str} to {date_to_str}',
            action_flag=ADDITION,
            change_message=f'{len(prediction_objects)} predictions made for period {date_from_str} to {date_to_str}'
        )

        messages.success(
            request, 
            f'Prediction completed! {len(prediction_objects)} medicines analyzed.'
        )
        return redirect('predictPage')

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        messages.error(request, f'Prediction error: {str(e)}')
        return redirect('predictPage')



#result views
@login_required
def resultView(request):
    # Get all predictions for current user
    predictions_list = Prediction.objects.filter(user=request.user).order_by('-created_at')
    
    # Get unique values for filters (excluding empty/null values)
    medicines = predictions_list.filter(medicine_name__isnull=False)\
        .exclude(medicine_name='')\
        .values_list('medicine_name', flat=True)\
        .distinct()\
        .order_by('medicine_name')
    
    brands = predictions_list.filter(brand_name__isnull=False)\
        .exclude(brand_name='')\
        .values_list('brand_name', flat=True)\
        .distinct()\
        .order_by('brand_name')
    
    categories = predictions_list.filter(category__isnull=False)\
        .exclude(category='')\
        .values_list('category', flat=True)\
        .distinct()\
        .order_by('category')
    
    # Get unique demand levels from predicted_demand field
    demand_levels_raw = predictions_list.filter(predicted_demand__isnull=False)\
        .exclude(predicted_demand='')\
        .values_list('predicted_demand', flat=True)\
        .distinct()
    
    # Sort demand levels in logical order: High, Medium, Low
    demand_level_order = {'High': 1, 'Medium': 2, 'Low': 3}
    demand_levels = sorted(
        [dl for dl in demand_levels_raw if dl in demand_level_order],
        key=lambda x: demand_level_order.get(x, 99)
    )
    
    # Get distinct date ranges (date_from to date_to)
    date_ranges = predictions_list.values_list('date_from', 'date_to').distinct()
    
    # Format date ranges for dropdown - ensure they are distinct
    unique_date_ranges = set()
    formatted_date_ranges = []
    
    for date_from, date_to in date_ranges:
        # Create a unique identifier for the date range (for filtering)
        range_str = f"{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}"
        if range_str not in unique_date_ranges:
            unique_date_ranges.add(range_str)
            # Format for display: "Jan 01, 2025 to Jan 31, 2025"
            display_str = f"{date_from.strftime('%b %d, %Y')} to {date_to.strftime('%b %d, %Y')}"
            formatted_date_ranges.append({
                'value': range_str,
                'display': display_str,
                'date_from': date_from  # Keep for sorting
            })
    
    # Sort date ranges by date_from (most recent/latest first)
    formatted_date_ranges.sort(key=lambda x: x['date_from'], reverse=True)
    
    context = {
        'predictions': predictions_list,
        'medicines': medicines,
        'brands': brands,
        'categories': categories,
        'demand_levels': demand_levels,
        'date_ranges': [dr['value'] for dr in formatted_date_ranges],
        'date_ranges_display': formatted_date_ranges,
    }
    
    return render(request, 'result.html', context)



def is_admin(user):
    """Check if user is admin or superuser"""
    return user.is_superuser or user.role == 'admin'


@login_required
@user_passes_test(is_admin)
def manageUserView(request):
    """Display user management page with users and activity logs"""
    users = User.objects.all().order_by('-created_at')
    logs = LogEntry.objects.select_related('user', 'content_type').order_by('-action_time')[:50]
    
    context = {
        'users': users,
        'logs': logs,
    }
    return render(request, 'manage-user.html', context)

@login_required
@user_passes_test(is_admin)
def addUserView(request):
    """Handle adding a new user"""
    if request.method == 'POST':
        username = request.POST.get('username')
        firstname = request.POST.get('firstname')
        lastname = request.POST.get('lastname')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        role = request.POST.get('role', 'pharmacist')
        
        # Validation
        if password != password_confirm:
            messages.error(request, 'Passwords do not match.')
            return redirect('manageUsersPage')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('manageUsersPage')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists.')
            return redirect('manageUsersPage')
        
        # Create user
        try:
            user = User.objects.create_user(
                username=username,
                first_name=firstname,
                last_name=lastname,
                email=email,
                password=password,
                role=role
            )
            
            # Create audit log entry
            LogEntry.objects.create(
                user_id=request.user.id,
                content_type_id=ContentType.objects.get_for_model(User).pk,
                object_id=user.id,
                object_repr=f'{user.username} ({user.email})',
                action_flag=ADDITION,
                change_message=f'Created user "{username}" with role "{role}"'
            )
            
            messages.success(request, f'User {username} created successfully.')
        except Exception as e:
            messages.error(request, f'Error creating user: {str(e)}')
        
    return redirect('manageUsersPage')


@login_required
@user_passes_test(is_admin)
def editUserView(request):
    """Handle editing an existing user"""
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        user = get_object_or_404(User, pk=user_id)
        
        # Prevent editing superusers (unless you are one)
        if user.is_superuser and not request.user.is_superuser:
            messages.error(request, 'You cannot edit superuser accounts.')
            return redirect('manageUsersPage')
        
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        role = request.POST.get('role')
        is_active = request.POST.get('is_active') == 'on'
        
        # Store old values for change tracking
        old_username = user.username
        old_email = user.email
        old_role = user.role
        old_is_active = user.is_active
        
        # Check for duplicate username (excluding current user)
        if User.objects.filter(username=username).exclude(pk=user_id).exists():
            messages.error(request, 'Username already exists.')
            return redirect('manageUsersPage')
        
        # Check for duplicate email (excluding current user)
        if User.objects.filter(email=email).exclude(pk=user_id).exists():
            messages.error(request, 'Email already exists.')
            return redirect('manageUsersPage')
        
        try:
            user.username = username
            user.email = email
            user.is_active = is_active
            
            # Don't change role for superusers
            if not user.is_superuser:
                user.role = role
            
            # Update password only if provided
            password_changed = False
            if password:
                user.set_password(password)
                password_changed = True
            
            user.save()
            
            # Build change message
            changes = []
            if old_username != username:
                changes.append(f'username: "{old_username}" → "{username}"')
            if old_email != email:
                changes.append(f'email: "{old_email}" → "{email}"')
            if not user.is_superuser and old_role != role:
                changes.append(f'role: "{old_role}" → "{role}"')
            if old_is_active != is_active:
                changes.append(f'status: {"active" if old_is_active else "inactive"} → {"active" if is_active else "inactive"}')
            if password_changed:
                changes.append('password changed')
            
            change_message = ', '.join(changes) if changes else 'No changes made'
            
            # Create audit log entry
            LogEntry.objects.create(
                user_id=request.user.id,
                content_type_id=ContentType.objects.get_for_model(User).pk,
                object_id=user.id,
                object_repr=f'{user.username} ({user.email})',
                action_flag=CHANGE,
                change_message=f'Updated user: {change_message}'
            )
            
            messages.success(request, f'User {username} updated successfully.')
        except Exception as e:
            messages.error(request, f'Error updating user: {str(e)}')
    
    return redirect('manageUsersPage')


@login_required
@user_passes_test(is_admin)
def deleteUserView(request):
    """Handle deleting a user and all associated data"""
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        user = get_object_or_404(User, pk=user_id)
        
        # Prevent deleting superusers
        if user.is_superuser:
            messages.error(request, 'Cannot delete superuser accounts.')
            return redirect('manageUsersPage')
        
        # Prevent self-deletion
        if user == request.user:
            messages.error(request, 'You cannot delete your own account.')
            return redirect('manageUsersPage')
        
        try:
            import shutil
            
            username = user.username
            user_email = user.email
            user_role = user.role
            user_id_str = str(user.id)
            deleted_user_id = user.id
            
            # Delete user's uploaded files
            user_upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', user_id_str)
            if os.path.exists(user_upload_dir):
                shutil.rmtree(user_upload_dir)
            
            # Delete user's trained models
            user_model_dir = os.path.join(settings.MEDIA_ROOT, 'models', user_id_str)
            if os.path.exists(user_model_dir):
                shutil.rmtree(user_model_dir)
            
            # Delete user (cascades to Upload, MedicineData, PredictionResult, etc.)
            user.delete()
            
            # Create audit log entry
            LogEntry.objects.create(
                user_id=request.user.id,
                content_type_id=ContentType.objects.get_for_model(User).pk,
                object_id=deleted_user_id,
                object_repr=f'{username} ({user_email})',
                action_flag=DELETION,
                change_message=f'Deleted user "{username}" (role: {user_role})'
            )
            
            messages.success(request, f'User {username} and all associated data deleted successfully.')
        except Exception as e:
            messages.error(request, f'Error deleting user: {str(e)}')
    
    return redirect('manageUsersPage')

@login_required
def searchView(request):
    return render(request, 'search.html')




@login_required
def searchMed(request):
    search_term = ''
    medicines = []
    monthly_summary = []
    total_unique_count = 0
    
    if request.method == 'POST':
        search_term = request.POST.get('usage', '').strip()
        
        if search_term:
            # Search in medicine usage field (case-insensitive)
            medicine_records = MedicineData.objects.filter(
                usage__icontains=search_term
            )
            
            # Get all distinct combinations from database
            raw_distinct = medicine_records.values(
                'medicine_name',
                'brand_name',
                'category'
            ).distinct()
            
            # Deduplicate using Python with normalized keys
            # Process ALL results first, then limit for display
            seen = set()
            unique_medicines = []
            
            for med in raw_distinct:
                # Create normalized key (lowercase, stripped)
                brand = (med['brand_name'] or '').strip()
                name = (med['medicine_name'] or '').strip()
                category = (med['category'] or '').strip()
                
                key = (
                    brand.lower(),
                    name.lower(),
                    category.lower()
                )
                
                if key not in seen:
                    seen.add(key)
                    unique_medicines.append({
                        'brand_name': brand,
                        'medicine_name': name,
                        'category': category
                    })
            
            # Store total count
            total_unique_count = len(unique_medicines)
            
            # Return all unique medicines (template pagination handles display)
            medicines_data = []
            
            for med in unique_medicines:
                medicine_name = med['medicine_name']
                brand_name = med['brand_name']
                category = med['category']
                
                # Get records for this medicine using case-insensitive matching
                # Also handle empty/null values properly
                if brand_name:
                    med_records = medicine_records.filter(
                        medicine_name__iexact=medicine_name,
                        brand_name__iexact=brand_name
                    )
                else:
                    med_records = medicine_records.filter(
                        medicine_name__iexact=medicine_name,
                        brand_name__isnull=True
                    ) | medicine_records.filter(
                        medicine_name__iexact=medicine_name,
                        brand_name__exact=''
                    )
                
                # Get usage from first record
                usage = med_records.values_list('usage', flat=True).first()
                
                # Calculate aggregates
                aggregates = med_records.aggregate(
                    total_qty=Sum('qty'),
                    avg_qty=Avg('qty'),
                    record_count=Count('id')
                )
                
                # Get monthly data for this medicine (most recent 12 months)
                monthly_data_qs = med_records.annotate(
                    month=TruncMonth('date')
                ).values(
                    'month'
                ).annotate(
                    total_qty=Sum('qty'),
                    year=ExtractYear('date'),
                    month_num=ExtractMonth('date')
                ).order_by('-month')[:12]  # Get most recent 12, descending
                
                # Reverse to chronological order (Jan first)
                monthly_data = list(monthly_data_qs)[::-1]
                
                # Format monthly data
                formatted_monthly = []
                month_names = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                
                for month in monthly_data:
                    formatted_monthly.append({
                        'month_name': month_names.get(month['month_num'], 'Unknown'),
                        'year': month['year'],
                        'total_qty': month['total_qty'] or 0
                    })
                
                medicines_data.append({
                    'medicine_name': medicine_name or 'Unknown',
                    'brand_name': brand_name or 'Unknown',
                    'category': category or 'Uncategorized',
                    'usage': usage or '',
                    'total_qty': aggregates['total_qty'] or 0,
                    'avg_qty': aggregates['avg_qty'] or 0,
                    'record_count': aggregates['record_count'] or 0,
                    'monthly_data': formatted_monthly
                })
            
            medicines = medicines_data
            
            # Get overall monthly summary for the search term
            monthly_summary = list(medicine_records.annotate(
                month=TruncMonth('date')
            ).values(
                'month'
            ).annotate(
                total_qty=Sum('qty'),
                year=ExtractYear('date'),
                month_num=ExtractMonth('date')
            ).order_by('-month')[:12])
            
            # Format monthly summary
            month_names_full = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            for month in monthly_summary:
                month['month_name'] = f"{month_names_full.get(month['month_num'], 'Unknown')} {month['year']}"
    
    context = {
        'search_term': search_term,
        'medicines': medicines,
        'monthly_summary': monthly_summary,
        'total_unique_count': total_unique_count,  # Total unique medicines found
    }
    
    return render(request, 'search.html', context)



@login_required
def profileView(request):
    """Display user profile page"""
    return render(request, 'profile.html')

@login_required
def profileChange(request):
    """Handle profile information update"""
    if request.method == 'POST':
        user = request.user
        
        # Get form data
        first_name = request.POST.get('first_name', '').strip()
        last_name = request.POST.get('last_name', '').strip()
        email = request.POST.get('email', '').strip()
        
        # Store old values for change tracking
        old_first_name = user.first_name
        old_last_name = user.last_name
        old_email = user.email
        
        # Validation
        errors = []
        
        if not first_name:
            errors.append('First name is required.')
        
        if not last_name:
            errors.append('Last name is required.')
        
        if not email:
            errors.append('Email is required.')
        
        # Check if email is already taken by another user
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        if User.objects.exclude(pk=user.pk).filter(email=email).exists():
            errors.append('This email is already in use by another account.')
        
        # If there are errors, show them and redirect back
        if errors:
            for error in errors:
                messages.error(request, error)
            return redirect('profilePage')
        
        # Update user information
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.save()
        
        # Build change message
        changes = []
        if old_first_name != first_name:
            changes.append(f'first name: "{old_first_name}" → "{first_name}"')
        if old_last_name != last_name:
            changes.append(f'last name: "{old_last_name}" → "{last_name}"')
        if old_email != email:
            changes.append(f'email: "{old_email}" → "{email}"')
        
        change_message = ', '.join(changes) if changes else 'No changes made'
        
        # Create audit log entry
        LogEntry.objects.create(
            user_id=request.user.id,
            content_type_id=ContentType.objects.get_for_model(User).pk,
            object_id=user.id,
            object_repr=f'{user.username} ({user.email})',
            action_flag=CHANGE,
            change_message=f'Profile updated: {change_message}'
        )
        
        messages.success(request, 'Your profile has been updated successfully!')
        return redirect('profilePage')
    
    # If GET request, redirect to profile page
    return redirect('profilePage')


@login_required
def passwordChange(request):
    """Handle password change"""
    if request.method == 'POST':
        user = request.user
        
        old_password = request.POST.get('old_password', '')
        new_password1 = request.POST.get('new_password1', '')
        new_password2 = request.POST.get('new_password2', '')
        
        # Validation
        errors = []
        
        # Check if old password is correct
        if not user.check_password(old_password):
            errors.append('Your current password is incorrect.')
        
        # Check if new passwords match
        if new_password1 != new_password2:
            errors.append('New passwords do not match.')
        
        # Check password strength
        if len(new_password1) < 8:
            errors.append('Password must be at least 8 characters long.')
        
        if new_password1.isdigit():
            errors.append('Password cannot be entirely numeric.')
        
        if new_password1.lower() in ['password', '12345678', 'qwerty123']:
            errors.append('Password is too common. Please choose a stronger password.')
        
        # If there are errors, show them and redirect back
        if errors:
            for error in errors:
                messages.error(request, error)
            return redirect('profilePage')
        
        # Update password
        user.set_password(new_password1)
        user.save()
        
        # Keep user logged in after password change
        update_session_auth_hash(request, user)
        
        # Create audit log entry
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        LogEntry.objects.create(
            user_id=request.user.id,
            content_type_id=ContentType.objects.get_for_model(User).pk,
            object_id=user.id,
            object_repr=f'{user.username} ({user.email})',
            action_flag=CHANGE,
            change_message='Password changed by user'
        )
        
        messages.success(request, 'Your password has been changed successfully!')
        return redirect('profilePage')
    
    # If GET request, redirect to profile page
    return redirect('profilePage')




@login_required
def feedback_page(request):
    user_feedbacks = Feedback.objects.filter(user=request.user)[:5]
    
    context = {
        'user_feedbacks': user_feedbacks,
    }
    
    # Admin gets all feedback
    if request.user.is_superuser or request.user.role == 'admin':
        all_feedbacks = Feedback.objects.select_related('user').all()
        context['all_feedbacks'] = all_feedbacks
        context['excellent_count'] = all_feedbacks.filter(rating=5).count()
        context['good_count'] = all_feedbacks.filter(rating=4).count()
        context['low_count'] = all_feedbacks.filter(rating__lte=3).count()
    
    return render(request, 'feedback.html', context)


@login_required
def submit_feedback(request):
    if request.method == 'POST':
        subject = request.POST.get('subject', '').strip()
        message = request.POST.get('message', '').strip()
        rating = request.POST.get('rating', 5)
        
        if not subject or not message:
            messages.error(request, 'Subject and message are required.')
            return redirect('feedback_page')
        
        try:
            rating = int(rating)
            if rating < 1 or rating > 5:
                rating = 5
        except:
            rating = 5
        
        Feedback.objects.create(
            user=request.user,
            subject=subject,
            message=message,
            rating=rating
        )
        
        messages.success(request, 'Thank you for your feedback!')
        return redirect('feedback_page')
    
    return redirect('feedback_page')

@login_required
def delete_feedback(request, feedback_id):
    if not request.user.is_superuser and request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('feedback_page')
    
    feedback = get_object_or_404(Feedback, id=feedback_id)
    feedback.delete()
    messages.success(request, 'Feedback deleted successfully.')
    return redirect('feedback_page')