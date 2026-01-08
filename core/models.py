
# models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator


class User(AbstractUser):
    """Custom user model with role-based access"""
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('pharmacist', 'Pharmacist'),
    ]
    
    # Override email to make it required
    email = models.EmailField(unique=True)
    
    # Custom fields
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='pharmacist'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    
    # Fields inherited from AbstractUser (no need to redefine):
    # username, first_name, last_name, password, is_staff, is_active, etc.
    
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def save(self, *args, **kwargs):
        # Automatically set role to 'admin' for superusers
        if self.is_superuser:
            self.role = 'admin'
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"


class Upload(models.Model):
    """File upload tracking"""
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='uploads'
    )
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Upload'
        verbose_name_plural = 'Uploads'
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.filename} - {self.user.username}"


class MedicineData(models.Model):
    """Medicine sales data"""
    upload = models.ForeignKey(
        Upload,
        on_delete=models.CASCADE,
        related_name='medicine_records'
    )
    date = models.DateField()
    medicine_name = models.CharField(max_length=255, blank=True, null=True)
    category = models.CharField(max_length=255, blank=True, null=True)
    brand_name = models.CharField(max_length=255, blank=True, null=True)
    usage = models.CharField(max_length=255, blank=True, null=True)
    qty = models.FloatField(
        verbose_name='Quantity',
        validators=[MinValueValidator(0.0)],
        blank=True,
        null=True
    )
    
    class Meta:
        verbose_name = 'Medicine Data'
        verbose_name_plural = 'Medicine Data'
        ordering = ['-date']
        indexes = [
            models.Index(fields=['date', 'medicine_name']),
            models.Index(fields=['category']),
        ]
    
    def __str__(self):
        return f"{self.medicine_name or 'Unknown'} - {self.date}"


class ModelTrainingInfo(models.Model):
    """ML model training information (Successful Runs Only)"""
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='trained_models'
    )
    
    # Model Configuration
    model_type = models.CharField(max_length=100, blank=True, null=True)
    learning_rate = models.FloatField(blank=True, null=True)
    max_depth = models.IntegerField(blank=True, null=True)
    n_estimators = models.IntegerField(blank=True, null=True)
    subsample = models.FloatField(blank=True, null=True)
    colsample_bytree = models.FloatField(blank=True, null=True)
    gamma = models.FloatField(blank=True, null=True)
    min_child_weight = models.IntegerField(blank=True, null=True)
    reg_alpha = models.FloatField(blank=True, null=True)
    reg_lambda = models.FloatField(blank=True, null=True)
    
    # Results
    accuracy = models.FloatField(blank=True, null=True)
    training_time = models.FloatField(blank=True, null=True)
    data_points = models.IntegerField(blank=True, null=True)
    
    # Status (Locked to 'success')
    status = models.CharField(
        max_length=20,
        default='completed',

    )
    
    trained_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Model Training Info'
        verbose_name_plural = 'Model Training Info'
        ordering = ['-trained_at']
    
    def __str__(self):
        # Accuracy is multiplied by 100 for display if stored as a decimal (e.g., 0.95 -> 95%)
        acc_display = self.accuracy * 100 if self.accuracy and self.accuracy <= 1 else (self.accuracy or 0)
        return f"{self.model_type or 'Model'} - Success ({acc_display:.2f}%)"


class Prediction(models.Model):
    """Demand predictions"""
    model = models.ForeignKey(
        ModelTrainingInfo,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    medicine_name = models.CharField(max_length=255, blank=True, null=True)
    brand_name = models.CharField(max_length=255, blank=True, null=True)
    category = models.CharField(max_length=255, blank=True, null=True)
    total_sales = models.FloatField(blank=True, null=True)
    date_from = models.DateField()
    date_to = models.DateField()
    predicted_demand = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(
        validators=[MinValueValidator(0.0)],
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.medicine_name or 'Unknown'} - {self.predicted_demand}"

