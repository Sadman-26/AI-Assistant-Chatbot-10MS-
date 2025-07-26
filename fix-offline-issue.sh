#!/bin/bash

echo "🔧 Fixing 'Offline' Issue - Step by Step"
echo "=========================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo ""
echo "📦 Step 1: Deploying Backend..."
echo "--------------------------------"

cd Backend

# Deploy backend
echo "Deploying backend to Vercel..."
BACKEND_URL=$(vercel --prod --yes 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)

if [ -z "$BACKEND_URL" ]; then
    echo "❌ Failed to get backend URL. Please deploy manually:"
    echo "cd Backend && vercel"
    exit 1
fi

echo "✅ Backend deployed to: $BACKEND_URL"

echo ""
echo "🧪 Step 2: Testing Backend..."
echo "-------------------------------"

# Test backend health
echo "Testing backend health endpoint..."
HEALTH_RESPONSE=$(curl -s "$BACKEND_URL/health" 2>/dev/null)

if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo "✅ Backend is working!"
else
    echo "❌ Backend health check failed. Response: $HEALTH_RESPONSE"
    echo "Please check your backend deployment."
    exit 1
fi

echo ""
echo "🌐 Step 3: Deploying Frontend..."
echo "---------------------------------"

cd ../Frontend

# Update API URL temporarily for testing
echo "Updating API URL to backend..."
sed -i "s|const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';|const API_BASE_URL = '$BACKEND_URL';|" src/services/api.ts

# Deploy frontend
echo "Deploying frontend to Vercel..."
FRONTEND_URL=$(vercel --prod --yes 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)

if [ -z "$FRONTEND_URL" ]; then
    echo "❌ Failed to get frontend URL. Please deploy manually:"
    echo "cd Frontend && vercel"
    exit 1
fi

echo "✅ Frontend deployed to: $FRONTEND_URL"

echo ""
echo "🎉 Step 4: Setup Complete!"
echo "---------------------------"
echo ""
echo "🔗 Your URLs:"
echo "   Frontend: $FRONTEND_URL"
echo "   Backend:  $BACKEND_URL"
echo ""
echo "📋 Next Steps:"
echo "1. Visit your frontend URL"
echo "2. Open Developer Tools (F12)"
echo "3. Check Console for debug messages"
echo "4. The status should show 'Online'"
echo ""
echo "🔧 For permanent fix:"
echo "1. Go to Vercel dashboard"
echo "2. Add environment variable: VITE_API_BASE_URL = $BACKEND_URL"
echo "3. Revert the API URL change in code"
echo ""
echo "📖 For detailed troubleshooting, see TROUBLESHOOTING.md" 