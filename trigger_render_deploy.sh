#!/bin/bash
# AudioLab Render Deployment Trigger
# This script uses Render's GitHub integration to trigger deployment

echo "🚀 AudioLab Render Deployment"
echo "================================"

# Check if we're in the right directory
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml not found. Please run from the AudioLab root directory."
    exit 1
fi

echo "✅ Found render.yaml configuration"

# Verify GitHub repository
echo "📋 Repository information:"
git remote -v | head -2

# Check if render.yaml is properly configured
echo -e "\n📄 Render configuration preview:"
head -10 render.yaml

echo -e "\n🔧 Triggering deployment via GitHub push..."

# Create a small update to trigger Render deployment
echo "# AudioLab deployment trigger - $(date)" >> .render_deploy_trigger

# Add and commit the trigger
git add .render_deploy_trigger
git commit -m "trigger: Deploy AudioLab to Render.io

- Trigger automatic deployment via GitHub integration
- Services: web service, PostgreSQL, Redis
- Configuration: render.yaml with standard plan
- Expected URL: https://audiolab-api.onrender.com"

# Push to trigger Render deployment
echo "📤 Pushing to GitHub to trigger Render deployment..."
git push origin master

echo -e "\n✅ Deployment trigger sent!"
echo "🔍 To monitor deployment:"
echo "   1. Visit: https://dashboard.render.com/"
echo "   2. Look for 'audiolab-api' service"
echo "   3. Monitor build logs and deployment status"
echo "   4. Check health at: https://audiolab-api.onrender.com/health"

echo -e "\n⏱️  Expected timeline:"
echo "   - Service detection: 1-2 minutes"
echo "   - Build process: 8-12 minutes"
echo "   - First startup: 1-2 minutes"
echo "   - Total: ~10-15 minutes"

echo -e "\n🎯 Next steps:"
echo "   1. Check Render dashboard for service creation"
echo "   2. Monitor build logs for progress"
echo "   3. Test health endpoint when live"
echo "   4. Verify API documentation at /docs"